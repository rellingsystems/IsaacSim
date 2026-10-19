# SPDX-License-Identifier: Apache-2.0
"""Open a writable streaming session layer without changing the imported scene."""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from pathlib import Path


async def prepare_scene() -> None:
    """Load the requested USD, add viewer lighting, and leave the timeline stopped."""
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.rendering_manager import ViewportManager
    from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Sdf, UsdLux, UsdPhysics

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
    ViewportManager.set_camera_view(str(viewport.camera_path), eye=[2.0, -2.0, 1.5], target=[0.3, -0.1, 0.6])
    for _ in range(60):
        await app.next_update_async()
    result = {"status": "ready", "source": str(source), "session": str(session_path), "timeline": "stopped"}
    (run / "scene-ready.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Relling streaming scene ready: {session_path}", flush=True)


async def report_errors() -> None:
    """Keep asynchronous startup failures visible in the session log."""
    try:
        await prepare_scene()
    except Exception:  # noqa: BLE001 - report startup failure from the deferred Kit task
        traceback.print_exc()


asyncio.ensure_future(report_errors())
