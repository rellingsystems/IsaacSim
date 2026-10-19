# SPDX-License-Identifier: Apache-2.0
"""Run a bounded procedural cube drop using the source build's SimulationApp."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    """Require a falling cube to settle on a collider using GPU PhysX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
            # Return from close so evidence is written only after clean teardown.
            "fast_shutdown": False,
        }
    )
    try:
        # Kit must initialize before importing its extensions.
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import PhysxScene, SimulationManager
        from pxr import UsdGeom

        stage = stage_utils.create_new_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = PhysxScene("/World/PhysicsScene")
        scene.set_gravity([0, 0, -9.81])
        scene.set_steps_per_second(60)
        scene.set_enabled_gpu_dynamics(True)
        scene.set_broadphase_type("GPU")
        SimulationManager.set_device(f"cuda:{args.gpu}")
        Cube("/World/Floor", sizes=1.0, scales=[10, 10, 0.1], positions=[0, 0, -0.05])
        GeomPrim("/World/Floor", apply_collision_apis=True)
        Cube("/World/Cube", sizes=0.5, positions=[0, 0, 2.0], colors="#00aaff")
        GeomPrim("/World/Cube", apply_collision_apis=True)
        rigid = RigidPrim("/World/Cube", masses=[1.0])
        DistantLight("/World/Light").set_intensities(1000)
        app_utils.play()
        app.update()
        heights = []
        for step in range(240):
            SimulationManager.step()
            if step % 30 == 0:
                RenderingManager.render()
            positions, _ = rigid.get_world_poses()
            heights.append(float(positions.numpy()[0, 2]))
        final_height = heights[-1]
        settled_range = max(heights[-30:]) - min(heights[-30:])
        if not all(math.isfinite(height) for height in heights):
            raise RuntimeError("Nonfinite physics position")
        if min(heights) >= 1.0 or not 0.20 <= final_height <= 0.30 or settled_range > 0.03:
            raise RuntimeError(f"Cube drop failed: final={final_height}, settled range={settled_range}")
        result = {
            "status": "passed",
            "steps": 240,
            "physics_dt_seconds": 1.0 / 60.0,
            "requested_physics_device": f"cuda:{args.gpu}",
            "initial_height_m": 2.0,
            "final_height_m": final_height,
            "settled_range_m": settled_range,
            "render_calls": 8,
        }
        app_utils.stop()
    finally:
        app.close()
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
