#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start, inspect, or stop this checkout's persistent private streaming process."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "_relling_runs"
STATE = RUNS / "stream-active.json"


def is_running(state: dict) -> bool:
    """Only recognize the Kit streaming process owned by this record."""
    command = Path(f"/proc/{state.get('pid', 0)}/cmdline")
    try:
        arguments = command.read_bytes()
    except OSError:
        return False
    return b"isaacsim.exp.full.streaming.kit" in arguments and str(ROOT).encode() in arguments


def main() -> None:
    """Keep viewer sessions independent of the initiating SSH connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--ip", help="Stallion's Tailscale IPv4 address")
    args = parser.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    running = is_running(state)
    if args.action == "status":
        print(json.dumps({**state, "running": running}, indent=2))
        return
    if args.action == "stop":
        if running:
            os.killpg(state["pid"], signal.SIGTERM)
            for _ in range(20):
                if not is_running(state):
                    break
                time.sleep(0.5)
            if is_running(state):
                os.killpg(state["pid"], signal.SIGKILL)
        print(json.dumps({"stop_requested": running, "run_dir": state.get("run_dir")}))
        return
    if running:
        raise RuntimeError("This checkout already has an active streaming session; inspect status or stop it first")
    args.scene = args.scene or (Path(state["scene"]) if state.get("scene") else None)
    args.ip = args.ip or state.get("ip")
    if not args.scene or not args.scene.is_file() or not args.ip:
        parser.error("Start requires an existing --scene and --ip")
    if ipaddress.ip_address(args.ip) not in ipaddress.ip_network("100.64.0.0/10"):
        parser.error("Use the host's private Tailscale IPv4 address")
    guard = ["sudo", "-n", "nft"]
    if subprocess.run(
        guard + ["list", "table", "inet", "relling_isaac_stream"], capture_output=True, check=False
    ).returncode:
        rules = str(ROOT / "tools/relling/stream_guard.nft")
        subprocess.run(guard + ["-c", "-f", rules], check=True)
        subprocess.run(guard + ["-f", rules], check=True)
    for port in (49100, 8226):
        with socket.socket() as check:
            check.bind(("127.0.0.1", port))
    build = ROOT / "_build/linux-x86_64/release"
    libraries = sorted((build / "exts/isaacsim.pip.nv/pip_prebundle/nvidia").glob("*/lib"))
    if not libraries:
        raise RuntimeError("Build runtime libraries are missing")
    run = RUNS / f"{time.time_ns()}-stream"
    run.mkdir(parents=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries)) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["RELLING_STREAM_SCENE"] = str(args.scene.resolve())
    env["RELLING_STREAM_RUN"] = str(run)
    command = [
        "bash",
        str(build / "isaac-sim.streaming.sh"),
        "--no-ros-env",
        "--enable",
        "isaacsim.code_editor.python_server",
        "--/exts/isaacsim.code_editor.python_server/host=127.0.0.1",
        f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.ip}",
        "--/exts/omni.kit.livestream.app/primaryStream/targetFps=30",
        "--/exts/omni.services.livestream.session/quitOnSessionEnded=false",
        "--/app/window/width=1600",
        "--/app/window/height=900",
        "--/renderer/multiGpu/enabled=false",
        "--/isaac/startup/ros_bridge_extension=",
        "--/isaac/startup/ros_sim_control_extension=false",
        "--/app/runLoops/main/rateLimitEnabled=true",
        "--/app/runLoops/main/rateLimitFrequency=30",
    ]
    with (run / "stream.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    state = {
        "pid": process.pid,
        "run_dir": str(run),
        "ip": args.ip,
        "scene": str(args.scene.resolve()),
        "command": command,
    }
    STATE.write_text(json.dumps(state, indent=2) + "\n")
    (run / "launch.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Streaming process exited; inspect {run / 'stream.log'}")
        with socket.socket() as check:
            if check.connect_ex(("127.0.0.1", 8226)) == 0:
                break
        time.sleep(0.5)
    else:
        raise TimeoutError("Local scene-control endpoint did not start")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "skills/isaac-sim-remote/scripts/isaacsim_send.py"),
            "--context",
            "relling-startup",
            "--file",
            str(ROOT / "tools/relling/stream_scene.py"),
            "--timeout",
            "30",
        ],
        check=True,
    )
    for _ in range(120):
        if (run / "scene-ready.json").exists():
            print((run / "scene-ready.json").read_text())
            return
        if process.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError(f"Scene did not become ready; inspect {run / 'stream.log'}")


if __name__ == "__main__":
    main()
