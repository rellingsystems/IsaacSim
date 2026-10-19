#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan against a frozen Giza snapshot and preview passing paths in the stopped USD viewer."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    """Publish a complete result atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan(snapshot: Path, worktree: Path, run_path: Path, output: Path) -> None:
    """Use robot-research's planner without connecting to a controller or changing its files."""
    import mujoco
    import numpy as np

    snapshot, worktree, output = snapshot.resolve(), worktree.resolve(), output.resolve()
    sys.path.insert(0, str(worktree))
    from planning.cartesian_path import CartesianPath
    from planning.planner import Planner

    output.mkdir(parents=True, exist_ok=False)
    (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    provenance = json.loads((snapshot / "provenance.json").read_text())
    source = snapshot / "scene.xml"
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    if identity != provenance["source_sha256"]:
        raise ValueError("Snapshot source hash differs from provenance")
    root = Path(__file__).resolve().parents[2]
    stream = json.loads((root / "_relling_runs/stream-active.json").read_text())
    if Path(stream.get("home", "")).resolve() != (snapshot / "home.json").resolve():
        raise ValueError("The viewer must use this exact source snapshot's home pose")
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    robot, study = provenance["robot"], provenance["resolved_study"]
    settings = {
        **study["planner"],
        "clearance_m": study["screen"]["clearance_m"],
        "distance_search_m": study["screen"]["distance_search_m"],
        "tool_site": robot["tool_site"],
        "robot_root": robot["root"],
        "joints": robot["joints"],
        "home_key": "home",
        "tool_quat_wxyz": "home",
    }
    run = json.loads(run_path.read_text())
    write_json(output / "request.json", run)
    started = time.monotonic()
    report, path = Planner.plan_path(model, data, settings, run)
    report.update(
        source=str(source),
        source_sha256=identity,
        source_commit=provenance["commit"],
        planner_commit=subprocess.check_output(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True).strip(),
        settings=settings,
        preview_type="kinematic; no dynamics, speed limits, force control, or hardware execution",
    )
    write_json(output / "result.json", report)
    if not report["passed"]:
        print(json.dumps(report, indent=2), flush=True)
        return
    # A separate checker revisits every edge at twice the original joint-space resolution.
    audit = CartesianPath(model, data, {**settings, "joint_step_rad": settings["joint_step_rad"] / 2})
    minimum, checked = settings["distance_search_m"], 0
    for index, q in enumerate(path):
        samples = [q] if not index else audit.joint_samples(path[index - 1], q)[1:]
        for sample in samples:
            check = audit.check(sample)
            checked += 1
            minimum = min(minimum, check.get("distance_m", 0))
            if not check["ok"]:
                report.update(passed=False, audit_failure={"path_index": index, **check})
                write_json(output / "result.json", report)
                raise ValueError(f"Denser collision audit failed: {check}")
    if np.max(np.abs(path[-1] - path[0])) > 1e-5:
        raise ValueError("Preview paths must return to their verified initial home configuration")
    frames = []
    for q in path:
        data.qpos[audit.qids] = q
        mujoco.mj_forward(model, data)
        frames.append(
            {
                "bodies": {
                    model.body(i).name: {"position": data.xpos[i].tolist(), "quaternion": data.xquat[i].tolist()}
                    for i in range(1, model.nbody)
                },
                "joints": {name: float(value) for name, value in zip(robot["joints"], q, strict=True)},
                "tool_position_m": data.site_xpos[audit.site].tolist(),
            }
        )
    report.update(
        audit_samples=checked,
        audit_min_environment_clearance_m=minimum,
        clearance_is_search_bound=minimum >= settings["distance_search_m"],
        seconds=time.monotonic() - started,
        trajectory_frames=len(frames),
    )
    write_json(output / "result.json", report)
    write_json(
        output / "trajectory.json",
        {
            "passed": True,
            "source_sha256": identity,
            "usd_source": stream["scene"],
            "home": stream["home"],
            "report": str(output / "result.json"),
            "frames": frames,
        },
    )
    print(json.dumps(report, indent=2), flush=True)


async def preview(trajectory: Path, repeats: int) -> None:
    """Replay verified source-body poses with physics stopped, then restore the first frame."""
    import numpy as np
    import omni.kit.app
    import omni.timeline
    import omni.usd
    import usdex.core
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    document = json.loads(trajectory.read_text())
    report = json.loads(Path(document["report"]).read_text())
    home = Path(document["home"])
    provenance = json.loads((home.parent / "provenance.json").read_text())
    if not document["passed"] or not report["passed"] or report["source_sha256"] != document["source_sha256"]:
        raise ValueError("Only a passing audited trajectory can be previewed")
    if provenance["source_sha256"] != document["source_sha256"]:
        raise ValueError("Trajectory source does not match its home snapshot")
    stage = omni.usd.get_context().get_stage()
    if document["usd_source"] not in stage.GetRootLayer().subLayerPaths:
        raise ValueError("Active USD differs from the planned scene")
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_stopped():
        raise ValueError("Stop the timeline before a kinematic preview")
    root = stage.GetDefaultPrim()
    links = {
        stage.GetPrimAtPath(p).GetName(): stage.GetPrimAtPath(p)
        for p in root.GetRelationship("isaac:physics:robotLinks").GetTargets()
    }
    joints = {
        stage.GetPrimAtPath(p).GetName(): stage.GetPrimAtPath(p)
        for p in root.GetRelationship("isaac:physics:robotJoints").GetTargets()
    }
    frames = document["frames"]
    ordered = sorted(
        frames[0]["bodies"], key=lambda name: links[usdex.core.getValidPrimName(name)].GetPath().pathElementCount
    )
    result = {"status": "running", "kind": "kinematic_preview", "frames": len(frames), "repeats": repeats}
    preview_run = trajectory.parent / f"{time.time_ns()}-preview"
    preview_run.mkdir()
    result_path = preview_run / "preview.json"
    write_json(trajectory.parent / "preview-active.json", {"result": str(result_path)})
    result["result"] = str(result_path)
    write_json(result_path, result)
    print(json.dumps(result), flush=True)
    worst = np.zeros(2)

    def read_errors(frame: dict) -> np.ndarray:
        cache = UsdGeom.XformCache()
        errors = np.zeros(2)
        for name in ordered:
            pose = frame["bodies"][name]
            matrix = cache.GetLocalToWorldTransform(links[usdex.core.getValidPrimName(name)])
            w, x, y, z = pose["quaternion"]
            expected = Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            errors = np.maximum(
                errors,
                [
                    np.max(np.abs(np.asarray(matrix.ExtractTranslation()) - pose["position"])),
                    np.max(np.abs(np.asarray(matrix.ExtractRotationMatrix()) - np.asarray(expected))),
                ],
            )
        if not np.isfinite(errors).all() or np.max(errors) > 2e-5:
            raise ValueError(f"USD preview pose errors {errors} exceed tolerance")
        return errors

    read_errors(frames[0])

    def apply_frame(frame: dict) -> np.ndarray:
        for name in ordered:
            pose = frame["bodies"][name]
            prim = links[usdex.core.getValidPrimName(name)]
            w, x, y, z = pose["quaternion"]
            world = Gf.Matrix4d(Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z))), Gf.Vec3d(*pose["position"]))
            parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * parent.GetInverse())
        for name, value in frame["joints"].items():
            prim = joints[usdex.core.getValidPrimName(name)]
            native = value * 180 / 3.141592653589793
            PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr(native)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            if drive:
                drive.CreateTargetPositionAttr(native)
        return read_errors(frame)

    try:
        app = omni.kit.app.get_app()
        for _ in range(repeats):
            for frame in frames:
                if not timeline.is_stopped() or omni.usd.get_context().get_stage() != stage:
                    raise RuntimeError("Stage or timeline changed during preview")
                worst = np.maximum(worst, apply_frame(frame))
                await app.next_update_async()
        result.update(
            status="passed", max_body_position_error_m=float(worst[0]), max_rotation_matrix_error=float(worst[1])
        )
    except asyncio.CancelledError:
        result.update(status="stopped")
    except Exception as exc:  # noqa: BLE001 - publish asynchronous preview failures
        result.update(status="failed", error=str(exc))
        traceback.print_exc()
    finally:
        if omni.usd.get_context().get_stage() == stage:
            timeline.stop()
            result["restored_home_position_error_m"] = float(apply_frame(frames[0])[0])
        write_json(result_path, result)
        print(json.dumps(result), flush=True)


def main() -> None:
    """Dispatch offline planning or loopback preview control."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--snapshot", type=Path, required=True)
    planner.add_argument("--worktree", type=Path, required=True)
    planner.add_argument("--run", type=Path, required=True)
    planner.add_argument("--output", type=Path, required=True)
    player = commands.add_parser("preview")
    player.add_argument("--trajectory", type=Path, required=True)
    player.add_argument("--repeats", type=int, default=1, choices=range(1, 21))
    commands.add_parser("stop")
    args = parser.parse_args()
    if args.action == "plan":
        plan(args.snapshot, args.worktree, args.run, args.output)
        return
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "skills/isaac-sim-remote/scripts/isaacsim_send.py"),
        "--context",
        "relling-motion",
        "--file",
        str(Path(__file__).resolve()),
    ]
    command += (
        ["--arg", "stop_preview=True"]
        if args.action == "stop"
        else ["--arg", f"trajectory_path={args.trajectory.resolve()}", "--arg", f"preview_repeats={args.repeats}"]
    )
    subprocess.run(command, check=True)


if globals().get("stop_preview"):
    previous = globals().get("preview_task")
    if previous and not previous.done():
        previous.cancel()
    stop_preview = False
elif globals().get("trajectory_path"):
    previous = globals().get("preview_task")
    if previous and not previous.done():
        raise RuntimeError("A preview is already running; stop it first")
    preview_task = asyncio.ensure_future(preview(Path(globals()["trajectory_path"]), globals()["preview_repeats"]))
    trajectory_path = None
elif __name__ == "__main__":
    main()
