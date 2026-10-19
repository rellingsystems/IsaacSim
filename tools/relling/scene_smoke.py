# SPDX-License-Identifier: Apache-2.0
"""Load one exported scene and run a bounded headless simulation without controllers."""

from __future__ import annotations

import argparse
import faulthandler
import json
import traceback
from pathlib import Path


def main() -> None:
    """Verify that a composed USD scene initializes, steps, and renders with finite body poses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.scene.is_file() or args.output.exists():
        parser.error("Scene must exist and output must be new")
    faulthandler.dump_traceback_later(120, repeat=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False})
    exit_code = 0
    try:
        import numpy as np
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
        from pxr import Usd, UsdPhysics

        opened, stage = stage_utils.open_stage(str(args.scene.resolve()))
        if not opened or stage is None:
            raise RuntimeError("Failed to open exported stage")
        prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
        scene_paths = [str(prim.GetPath()) for prim in prims if prim.IsA(UsdPhysics.Scene)]
        physics = PhysxScene(scene_paths[0] if scene_paths else "/RellingSmoke/PhysicsScene")
        physics.set_gravity([0, 0, -9.81])
        physics.set_steps_per_second(60)
        physics.set_enabled_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
        SimulationManager.set_device("cuda:0")
        body_paths = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not body_paths:
            raise RuntimeError("Scene has no rigid bodies for a dynamic smoke test")
        bodies = RigidPrim(body_paths)
        app_utils.play()
        app.update()
        for step in range(60):
            SimulationManager.step()
            if step % 15 == 0:
                RenderingManager.render()
            positions, orientations = bodies.get_world_poses()
            if not np.isfinite(positions.numpy()).all() or not np.isfinite(orientations.numpy()).all():
                raise RuntimeError("Nonfinite rigid-body pose")
        app_utils.stop()
        result = {
            "status": "passed",
            "scene": str(args.scene.resolve()),
            "steps": 60,
            "render_calls": 4,
            "rigid_bodies": len(body_paths),
            "requested_physics_device": "cuda:0",
            "acceptance": "Scene initialized, stepped, rendered, and retained finite body poses",
            "dynamics_equivalence": "not_tested",
            "controllers": "not_loaded",
        }
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
