# SPDX-License-Identifier: Apache-2.0
"""Open a writable streaming session layer without changing the imported scene."""

from __future__ import annotations

import asyncio
import json
import math
import os
import traceback
from pathlib import Path


def apply_home(stage, home_path: Path) -> dict:
    """Transfer the verified source pose into the writable viewer layer."""
    import numpy as np
    import usdex.core
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    home = json.loads(home_path.read_text())
    root = stage.GetDefaultPrim()
    joints = {
        stage.GetPrimAtPath(target).GetName(): str(target)
        for target in root.GetRelationship("isaac:physics:robotJoints").GetTargets()
    }
    values = {}
    for name, value in home["joints"].items():
        identifier = usdex.core.getValidPrimName(name)
        if identifier not in joints:
            raise ValueError(f"Home joint {name!r} is missing from the imported robot")
        prim = stage.GetPrimAtPath(joints[identifier])
        if prim.IsA(UsdPhysics.RevoluteJoint):
            axis, native = "angular", math.degrees(value)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            axis, native = "linear", value
        else:
            raise ValueError(f"Unsupported home joint type: {name!r}")
        PhysxSchema.JointStateAPI.Apply(prim, axis).CreatePositionAttr(native)
        drive = UsdPhysics.DriveAPI(prim, axis)
        if drive:
            drive.CreateTargetPositionAttr(native)
        values[joints[identifier]] = value
    links = {
        stage.GetPrimAtPath(target).GetName(): stage.GetPrimAtPath(target)
        for target in root.GetRelationship("isaac:physics:robotLinks").GetTargets()
    }
    poses = []
    for name, pose in home["bodies"].items():
        identifier = usdex.core.getValidPrimName(name)
        if identifier not in links:
            raise ValueError(f"Home body {name!r} is missing from the imported robot")
        poses.append((links[identifier], pose))
    # Parents precede children; read fresh authored USD transforms after each write.
    # Source poses come from MuJoCo's named home keyframe, not a second FK solver.
    for prim, pose in sorted(poses, key=lambda item: item[0].GetPath().pathElementCount):
        w, x, y, z = pose["quaternion"]
        world = Gf.Matrix4d(Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z))), Gf.Vec3d(*pose["position"]))
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        transform = UsdGeom.Xformable(prim)
        transform.MakeMatrixXform().Set(world * parent.GetInverse())
        transform.SetResetXformStack(False)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    positions, rotations = [], []
    for name, pose in home["bodies"].items():
        identifier = usdex.core.getValidPrimName(name)
        if identifier not in links:
            raise ValueError(f"Home body {name!r} is missing from the imported robot")
        matrix = cache.GetLocalToWorldTransform(links[identifier])
        positions.append(float(np.max(np.abs(np.asarray(matrix.ExtractTranslation()) - pose["position"]))))
        w, x, y, z = pose["quaternion"]
        expected = Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        rotations.append(float(np.max(np.abs(np.asarray(matrix.ExtractRotationMatrix()) - np.asarray(expected)))))
    result = {
        "method": "verified_mujoco_home_body_transforms",
        "bodies_checked": len(positions),
        "joints_applied": len(values),
        "max_position_error_m": max(positions),
        "max_rotation_matrix_error": max(rotations),
        "position_tolerance_m": 2e-5,
        "rotation_matrix_tolerance": 2e-5,
    }
    if result["max_position_error_m"] > 2e-5 or result["max_rotation_matrix_error"] > 2e-5:
        raise ValueError(f"USD home pose differs from MuJoCo: {result}")
    return result


async def prepare_scene() -> None:
    """Load the requested USD, add viewer lighting, and leave the timeline stopped."""
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.rendering_manager import ViewportManager
    from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    run = Path(os.environ["RELLING_STREAM_RUN"])
    source = Path(os.environ["RELLING_STREAM_SCENE"])
    app = omni.kit.app.get_app()
    for _ in range(120):
        await app.next_update_async()
    source_layer = Sdf.Layer.FindOrOpen(str(source))
    session_path = run / "deburr-session.usda"
    layer = Sdf.Layer.CreateNew(str(session_path))
    layer.subLayerPaths = [str(source)]
    layer.defaultPrim = source_layer.defaultPrim
    for name in ("upAxis", "metersPerUnit"):
        if source_layer.pseudoRoot.HasInfo(name):
            layer.pseudoRoot.SetInfo(name, source_layer.pseudoRoot.GetInfo(name))
    layer.Save()
    opened, error = await omni.usd.get_context().open_stage_async(str(session_path))
    if not opened:
        raise RuntimeError(f"Failed to open streaming session: {error}")
    stage = omni.usd.get_context().get_stage()
    omni.timeline.get_timeline_interface().stop()
    home_check = (
        apply_home(stage, Path(os.environ["RELLING_STREAM_HOME"])) if os.environ.get("RELLING_STREAM_HOME") else None
    )
    light = UsdLux.DomeLight.Define(stage, "/RellingViewer/Light")
    light.CreateIntensityAttr(900)
    physics_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    physics = PhysxScene(physics_paths[0] if physics_paths else "/RellingViewer/PhysicsScene")
    physics.set_gravity([0, 0, -9.81])
    physics.set_steps_per_second(60)
    physics.set_enabled_gpu_dynamics(True)
    physics.set_broadphase_type("GPU")
    SimulationManager.set_device("cuda:0")
    stage.GetRootLayer().Save()
    for _ in range(60):
        await app.next_update_async()
    omni.timeline.get_timeline_interface().stop()
    viewport = get_active_viewport()
    eye, target = [2.0, -2.0, 1.5], [0.3, -0.1, 0.6]
    if home_check:
        bounds = Gf.Range3d()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Plane):
                bounds.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
        if bounds.IsEmpty() or not all(math.isfinite(value) for value in bounds.GetSize()):
            raise ValueError("Imported cell has no finite visible geometry to frame")
        target = list(bounds.GetMidpoint())
        extent = max(bounds.GetSize())
        eye = [target[index] + extent * offset for index, offset in enumerate((1.1, -1.4, 0.9))]
    ViewportManager.set_camera_view(str(viewport.camera_path), eye=eye, target=target)
    for _ in range(60):
        await app.next_update_async()
    result = {"status": "ready", "source": str(source), "session": str(session_path), "timeline": "stopped"}
    if home_check:
        result["home_check"] = home_check
    (run / "scene-ready.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Relling streaming scene ready: {session_path}", flush=True)


async def report_errors() -> None:
    """Keep asynchronous startup failures visible in the session log."""
    try:
        await prepare_scene()
    except Exception:  # noqa: BLE001 - report startup failure from the deferred Kit task
        traceback.print_exc()


asyncio.ensure_future(report_errors())
