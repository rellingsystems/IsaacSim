#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check, build, and smoke-test the Relling source checkout on a Linux GPU host."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def read_command(command: list[str]) -> str:
    """Capture a bounded diagnostic command, preserving command failures."""
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()


def check_host(action: str, gpu: int) -> dict:
    """Report readiness and reject unsupported hosts or an occupied target GPU."""
    errors = []
    report = {"hostname": platform.node(), "system": platform.system(), "architecture": platform.machine()}
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append("This workflow targets native Linux x86_64; use Stallion, not macOS or WSL.")
    if "microsoft" in platform.release().lower():
        errors.append("WSL is not supported by this source-build workflow.")
    report["free_disk_gib"] = round(shutil.disk_usage(ROOT).free / GIB, 1)
    if report["free_disk_gib"] < (100 if action in ("preflight", "build") else 10):
        errors.append("Insufficient disk: reserve 100 GiB for a source build, 10 GiB for a smoke run.")
    if platform.system() == "Linux":
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        report["available_ram_gib"] = round(int(memory["MemAvailable"].split()[0]) / 1024**2, 1)
        if report["available_ram_gib"] < 16:
            errors.append("Insufficient available RAM: reserve 16 GiB and use two build jobs on a 32 GiB host.")
    commands = {
        "git": ["git", "--version"],
        "git_lfs": ["git", "lfs", "version"],
        "gpu_inventory": [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    }
    if action in ("preflight", "build"):
        commands.update({"gcc": ["gcc", "-dumpversion"], "g++": ["g++", "-dumpversion"]})
    for name, command in commands.items():
        try:
            report[name] = read_command(command)
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f"{name}: {error}")
    for name in ("gcc", "g++"):
        if name in report and report[name].split(".")[0] != "11":
            errors.append(f"{name} must resolve to version 11 (use a private PATH shim if needed).")
    if "gpu_inventory" in report:
        rows = list(csv.reader(io.StringIO(report["gpu_inventory"]), skipinitialspace=True))
        selected = next((row for row in rows if row[0].strip() == str(gpu)), None)
        if selected is None:
            errors.append(f"GPU {gpu} does not exist.")
        else:
            report["selected_gpu"] = selected
            try:
                # Driver bookkeeping can make a nominal 16 GiB card report slightly less.
                if float(selected[3]) < 15360:
                    errors.append("Selected GPU has less than 16 GiB VRAM.")
                if float(selected[4]) > 2048 or float(selected[5]) > 20:
                    errors.append("Selected GPU is occupied; wait for existing jobs before running this workflow.")
            except ValueError:
                errors.append("GPU memory/utilization is unavailable; inspect nvidia-smi before proceeding.")
    report["errors"] = errors
    report["status"] = "failed" if errors else "passed"
    return report


def run_logged(command: list[str], log_path: Path, timeout: int) -> None:
    """Run a process group with a deadline and retain its complete log."""
    print(f"Running: {' '.join(command)}\nLog: {log_path}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return_code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    """Execute one explicit workflow stage and write immutable run evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "build", "smoke"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--run-dir", type=Path, help="New directory; existing paths are never overwritten")
    args = parser.parse_args()
    if args.gpu < 0 or args.jobs < 1:
        parser.error("GPU must be nonnegative and jobs must be positive")
    report = check_host(args.action, args.gpu)
    print(json.dumps(report, indent=2))
    if args.action == "preflight":
        return int(bool(report["errors"]))
    if report["errors"]:
        return 1
    if args.action == "build" and not (ROOT / ".eula_accepted").is_file():
        print("Run ./tools/eula_check.sh interactively to review and accept NVIDIA's terms first.", file=sys.stderr)
        return 1
    run_dir = args.run_dir or ROOT / "_relling_runs" / f"{time.time_ns()}-{args.action}"
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "pipeline.pid").write_text(f"{os.getpid()}\n")
    result = {"status": "failed", "action": args.action, "gpu": args.gpu, "run_dir": str(run_dir)}
    try:
        result["commit"] = read_command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        result["worktree_status"] = read_command(["git", "-C", str(ROOT), "status", "--porcelain"])
        if args.action == "build":
            run_logged(["git", "lfs", "install", "--local"], run_dir / "lfs-install.log", 60)
            run_logged(["git", "lfs", "pull"], run_dir / "lfs-pull.log", 1800)
            run_logged(
                ["bash", "./build.sh", "--no-docker", "--release", "--jobs", str(args.jobs)],
                run_dir / "build.log",
                10800,
            )
            if not (ROOT / "_build/linux-x86_64/release/python.sh").is_file():
                raise FileNotFoundError("Build exited without producing the Python launcher")
        else:
            launcher = ROOT / "_build/linux-x86_64/release/python.sh"
            if not launcher.is_file():
                raise FileNotFoundError(f"Build the fork first; missing {launcher}")
            run_logged(
                [
                    "bash",
                    str(launcher),
                    str(ROOT / "tools/relling/smoke.py"),
                    "--output",
                    str(run_dir / "physics.json"),
                    "--gpu",
                    str(args.gpu),
                ],
                run_dir / "smoke.log",
                600,
            )
            physics = json.loads((run_dir / "physics.json").read_text())
            if physics.get("status") != "passed":
                raise RuntimeError("Physics acceptance failed")
            result["physics"] = physics
        result["status"] = "passed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        (run_dir / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
        (run_dir / "pipeline.pid").unlink(missing_ok=True)
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
