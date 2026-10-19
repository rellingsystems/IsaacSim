#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inventory and export existing MJCF catalogs through Isaac Sim's importer API."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    """Atomically update a progress record."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    """Read the content identity of a source XML."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(args: argparse.Namespace) -> None:
    """Classify every supplied path without loading a simulator or changing sources."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = {}
    excluded = collections.Counter()
    for name in args.paths.read_text().splitlines():
        source = Path(name).resolve()
        if str(source) in entries:
            entries[str(source)]["aliases"].append(name)
            continue
        try:
            tree = ET.parse(source)
            element = tree.getroot()
        except (OSError, ET.ParseError) as error:
            excluded[f"Unreadable or invalid XML: {type(error).__name__}"] += 1
            continue
        if element.tag not in ("mujoco", "mujocoinclude"):
            excluded["Other XML formats"] += 1
            continue
        relative = source.relative_to(args.source_base.resolve())
        identifier = hashlib.sha256(str(source).encode()).hexdigest()[:16]
        is_model = element.tag == "mujoco" and (
            element.find("worldbody") is not None or element.find("include") is not None
        )
        entry = {
            "id": identifier,
            "source": str(source),
            "relative_source": str(relative),
            "aliases": [name],
            "sha256": digest(source),
            "kind": "model" if is_model else "fragment",
            "xml_root": element.tag,
            "source_model": element.get("model"),
            "output": str(output / "assets" / relative.with_suffix("")),
        }
        entries[str(source)] = entry
        snapshot = output / "source_xml" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, snapshot)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_base": str(args.source_base.resolve()),
        "output_root": str(output),
        "scope": "MuJoCo XML from supplied current-project and vendor paths; physical aliases deduplicated",
        "limitations": [
            "Exports assets, not Python controllers, policies, task generators, or reward functions",
            "XML fragments are preserved but are not independent models",
            "Original files are read only; source_xml contains XML snapshots, not a full source mesh archive",
            "USD stage/dependency validation does not certify equivalent dynamics or physical robot operation",
        ],
        "excluded": dict(excluded),
        "entries": sorted(entries.values(), key=lambda entry: entry["source"]),
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "kinds": dict(collections.Counter(e["kind"] for e in entries.values())),
                "excluded": dict(excluded),
            },
            indent=2,
        )
    )


def resolve_resource(source: Path, directory: str, filename: str) -> Path:
    """Resolve an existing reference; relocate stale leading-parent paths only when unique."""
    reference = Path(directory) / filename
    normal = (source.parent / reference).resolve()
    if normal.is_file():
        return normal
    parts = list(reference.parts)
    while parts and parts[0] in ("..", "."):
        parts.pop(0)
    parents = list(source.parents)
    snapshots = ROOT / "_relling_source_snapshots"
    if source.is_relative_to(snapshots):
        snapshot = snapshots / source.relative_to(snapshots).parts[0]
        parents = [parent for parent in parents if parent.is_relative_to(snapshot)]
    candidates = {(parent / Path(*parts)).resolve() for parent in parents}
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        # Retain at least a directory and filename when relocating author-machine paths.
        for offset in range(1, len(parts) - 1):
            suffix = Path(*parts[offset:])
            candidates = {(parent / suffix).resolve() for parent in parents}
            existing = [path for path in candidates if path.is_file()]
            if existing:
                break
    if len(existing) != 1:
        raise FileNotFoundError(f"Resource {reference} has {len(existing)} unambiguous relocation candidates")
    return existing[0]


def prepare_source(source: Path, destination: Path) -> tuple[Path, list[dict], dict]:
    """Serialize a derived MJCF with absolute resources and supported legacy mesh files."""
    import mujoco
    import numpy as np
    from mujoco import msh2obj

    prepared = destination / "prepared"
    prepared.mkdir()
    spec = mujoco.MjSpec.from_file(str(source))
    changes = []
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        original = mesh.file
        resource = resolve_resource(source, mesh.compiler.meshdir, original)
        if resource.suffix.lower() == ".msh":
            output = prepared / f"{digest(resource)[:20]}.obj"
            if not output.exists():
                data = msh2obj.Msh.create(resource)
                text = msh2obj.msh_to_obj(resource)
                # MuJoCo's bundled writer incorrectly references absent UV/normal arrays.
                lines = []
                for line in text.splitlines():
                    if line.startswith("f "):
                        fields = []
                        for item in line.split()[1:]:
                            vertex, uv, normal = item.split("/")
                            fields.append(
                                vertex
                                + (
                                    ("/" + (uv if data.vertex_texcoords.size else "") + "/" + normal)
                                    if data.vertex_normals.size
                                    else ("/" + uv if data.vertex_texcoords.size else "")
                                )
                            )
                        line = "f " + " ".join(fields)
                    lines.append(line)
                output.write_text("\n".join(lines) + "\n")
            resource = output
            mesh.content_type = "model/obj"
        mesh.file = str(resource)
        changes.append({"kind": "mesh", "original": original, "resolved": str(resource)})
    for texture in spec.textures:
        if texture.file:
            original = texture.file
            texture.file = str(resolve_resource(source, texture.compiler.texturedir, original))
            changes.append({"kind": "texture", "original": original, "resolved": texture.file})
    for heightfield in spec.hfields:
        if heightfield.file:
            original = heightfield.file
            heightfield.file = str(resolve_resource(source, "", original))
            changes.append({"kind": "heightfield", "original": original, "resolved": heightfield.file})
    model = spec.compile()
    source_summary = {
        "bodies": model.nbody,
        "joints": model.njnt,
        "articulated_joints": int((model.jnt_type != mujoco.mjtJoint.mjJNT_FREE).sum()),
        "dofs": model.nv,
        "geometry": model.ngeom,
        "collision_geometry": int(((model.geom_contype | model.geom_conaffinity) != 0).sum()),
        "actuators": model.nu,
        "cameras": model.ncam,
    }
    result = prepared / source.name
    serialized = spec.to_xml()
    tree = ET.fromstring(serialized)
    if any(not node.get("class") for node in tree.findall(".//default/default")):
        # Attached models can serialize an unnamed nested default that MuJoCo cannot reload.
        # Rename default objects through their references; leave body/joint/geometry names intact.
        defaults = {spec.default}
        for group in (
            "bodies",
            "geoms",
            "joints",
            "sites",
            "actuators",
            "cameras",
            "lights",
            "meshes",
            "materials",
            "pairs",
            "equalities",
            "tendons",
        ):
            for element in getattr(spec, group, []):
                default = getattr(element, "classname", None)
                if default is not None and default not in defaults:
                    defaults.add(default)
                    default.name = f"relling_import_default_{len(defaults)}"
        spec.compile()
        serialized = spec.to_xml()
        roundtrip = mujoco.MjSpec.from_string(serialized).compile()
        for parameter in (
            "body_pos",
            "body_quat",
            "body_mass",
            "body_inertia",
            "geom_type",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_contype",
            "geom_conaffinity",
            "jnt_type",
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
        ):
            np.testing.assert_allclose(
                getattr(model, parameter), getattr(roundtrip, parameter), rtol=1e-5, atol=1e-6, err_msg=parameter
            )
        changes.append(
            {
                "kind": "attached_default_serialization",
                "default_objects": len(defaults),
                "check": "Compiled geometry, mass, inertia, joints and actuator parameters matched within XML precision",
                "relative_tolerance": 1e-5,
                "absolute_tolerance": 1e-6,
            }
        )
    result.write_text(serialized)
    write_json(prepared / "resource-references.json", changes)
    return result, changes, source_summary


def validate_stage(usd_path: Path, destination: Path, source_summary: dict) -> dict:
    """Select available physics and inspect composed, instanced geometry and dependencies."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError("Importer output cannot be opened")
    variants = stage.GetDefaultPrim().GetVariantSet("Physics")
    for selection in ("physx", "physics"):
        if selection in variants.GetVariantNames():
            variants.SetVariantSelection(selection)
            stage.GetRootLayer().Save()
            break
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))

    def is_dynamic_body(path: Sdf.Path) -> bool:
        prim = stage.GetPrimAtPath(path)
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(prim)
                return body.GetRigidBodyEnabledAttr().Get() is not False and not body.GetKinematicEnabledAttr().Get()
            prim = prim.GetParent()
        return False

    disabled_joints = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
        if targets and not any(is_dynamic_body(path) for path in targets):
            # PhysX rejects fixed joints between world/static/kinematic actors. Such a joint
            # adds no dynamic constraint and incorrectly anchors imported MuJoCo mocap markers.
            joint.CreateJointEnabledAttr(False)
            disabled_joints.append(str(prim.GetPath()))
    stage.GetRootLayer().Save()
    geometry = sum(prim.IsA(UsdGeom.Gprim) for prim in prims)
    if not geometry:
        raise RuntimeError("Imported stage contains no geometry")
    joint_count = sum(prim.IsA(UsdPhysics.Joint) for prim in prims)
    enabled_joint_count = sum(
        prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False for prim in prims
    )
    collision_count = sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in prims)
    if source_summary.get("articulated_joints", source_summary["joints"]) and not enabled_joint_count:
        raise RuntimeError("Source has joints but exported physics stage has none")
    if source_summary.get("collision_geometry", 0) and not collision_count:
        raise RuntimeError("Source has collision geometry but exported physics stage has none")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(usd_path)))
    dependencies = [layer.realPath for layer in layers if layer.realPath] + list(assets)
    external = [str(path) for path in dependencies if path and not Path(path).resolve().is_relative_to(destination)]
    if unresolved or external:
        raise RuntimeError(f"USD dependencies are incomplete or external: unresolved={unresolved}, external={external}")
    return {
        "usd_path": str(usd_path),
        "prim_count": len(prims),
        "geometry_count": geometry,
        "rigid_body_count": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "joint_count": joint_count,
        "enabled_joint_count": enabled_joint_count,
        "disabled_redundant_fixed_joints": disabled_joints,
        "collision_count": collision_count,
        "dependency_count": len(dependencies),
        "selected_physics_variant": variants.GetVariantSelection(),
    }


def audit_one(entry: dict, output: Path) -> dict:
    """Create a physics-selected entry layer while preserving original export records and layers."""
    import mujoco

    result = json.loads((output / "results" / f"{entry['id']}.json").read_text())
    source = Path(result["usd_path"])
    target = source.with_name(f"{source.stem}.isaac-{time.time_ns()}{source.suffix}")
    shutil.copy2(source, target)
    model = mujoco.MjModel.from_xml_path(result["prepared_source"])
    summary = dict(result["source_model_summary"])
    summary["articulated_joints"] = int((model.jnt_type != mujoco.mjtJoint.mjJNT_FREE).sum())
    summary["collision_geometry"] = int(((model.geom_contype | model.geom_conaffinity) != 0).sum())
    return {
        "status": "exported",
        "audit": "passed",
        "original_usd_path": str(source),
        "source_model_summary": summary,
        **validate_stage(target, Path(result["prepared_source"]).parent.parent, summary),
    }


def export_one(entry: dict) -> dict:
    """Convert one source without overrides and verify a nonempty, portable USD stage."""
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    source = Path(entry["source"])
    if digest(source) != entry["sha256"]:
        raise RuntimeError("Source XML changed after inventory")
    destination = Path(entry["output"])
    if destination.exists():
        destination = destination.with_name(f"{destination.name}_attempt_{time.time_ns()}")
    destination.mkdir(parents=True, exist_ok=False)
    license_sources = []
    for directory in source.parents:
        # Source snapshots live under this checkout; its code license does not license source assets.
        if directory == ROOT:
            break
        notices = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        ]
        if notices:
            (destination / "notices").mkdir()
            for notice in notices:
                shutil.copy2(notice, destination / "notices" / notice.name)
                license_sources.append(str(notice))
            break
    input_path, changes, source_summary = prepare_source(source, destination)
    if not source_summary["geometry"]:
        return {
            "status": "fragment",
            "reason": "Compiled MJCF contains no standalone geometry",
            "source_model_summary": source_summary,
            "source_sha256": entry["sha256"],
            "prepared_source": str(input_path),
        }
    importer = MJCFImporter(
        MJCFImporterConfig(
            mjcf_path=str(input_path),
            usd_path=str(destination),
            import_scene=True,
            fix_base=None,
            allow_self_collision=True,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
    )
    usd_path = Path(importer.import_mjcf()).resolve()
    validation = validate_stage(usd_path, destination, source_summary)
    if digest(source) != entry["sha256"]:
        raise RuntimeError("Source XML changed during export")
    result = {
        "status": "exported",
        **validation,
        "source_sha256": entry["sha256"],
        "physics_equivalence": "not_tested",
        "controller_port": "not_performed",
        "prepared_source": str(input_path),
        "resource_reference_count": len(changes),
        "source_model_summary": source_summary,
        "notice_sources": license_sources,
    }
    del importer
    gc.collect()
    return result


def worker(args: argparse.Namespace, manifest: dict) -> None:
    """Use one bounded Kit session for a small conversion chunk."""
    sys.argv = [sys.argv[0]]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0})
    try:
        import omni.kit.app
        import omni.usd

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
        manager.set_extension_enabled_immediate("isaacsim.robot.schema", True)
        for entry in manifest["entries"]:
            if entry["id"] not in args.ids:
                continue
            result = {"id": entry["id"], "source": entry["source"], "status": "failed"}
            start = time.monotonic()
            try:
                result.update(audit_one(entry, Path(manifest["output_root"])) if args.audit else export_one(entry))
            except Exception as error:  # noqa: BLE001 - isolate failures at the vendor model boundary
                result["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            result["seconds"] = round(time.monotonic() - start, 3)
            write_json(
                Path(manifest["output_root"]) / ("audits" if args.audit else "results") / f"{entry['id']}.json", result
            )
            print(json.dumps(result), flush=True)
            omni.usd.get_context().new_stage()
            gc.collect()
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


def summarize(manifest: dict) -> dict:
    """Build a complete ledger, including pending work and XML fragments."""
    output = Path(manifest["output_root"])
    entries = []
    for entry in manifest["entries"]:
        result_path = output / "results" / f"{entry['id']}.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {"status": "fragment" if entry["kind"] == "fragment" else "pending"}
        )
        audit_path = output / "audits" / f"{entry['id']}.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        entries.append({**entry, **result, **audit})
    counts = dict(collections.Counter(entry["status"] for entry in entries))
    summary = {"output_root": str(output), "counts": counts, "limitations": manifest["limitations"], "entries": entries}
    write_json(output / "catalog.json", summary)
    print(json.dumps({"output_root": str(output), "counts": counts}), flush=True)
    return summary


def run(args: argparse.Namespace, manifest: dict) -> None:
    """Supervise serial chunks, retaining progress across a crash or deadline."""
    output = Path(manifest["output_root"])
    (output / "results").mkdir(exist_ok=True)
    records = output / ("audits" if args.audit else "results")
    records.mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    if args.retry_failed or args.retry_exported:
        history = output / "results" / "history"
        history.mkdir(exist_ok=True)
        for record in (output / "results").glob("*.json"):
            result = json.loads(record.read_text())
            retry_statuses = ({"failed"} if args.retry_failed else set()) | (
                {"exported"} if args.retry_exported else set()
            )
            if result.get("status") in retry_statuses and (
                not args.match or any(term in result["source"] for term in args.match)
            ):
                record.replace(history / f"{record.stem}-{time.time_ns()}.json")
                previous_audit = output / "audits" / record.name
                if previous_audit.exists():
                    previous_audit.replace(history / f"audit-{record.stem}-{time.time_ns()}.json")
    pending = [
        entry
        for entry in manifest["entries"]
        if entry["kind"] == "model"
        and not (records / f"{entry['id']}.json").exists()
        and (
            not args.audit
            or (
                (output / "results" / f"{entry['id']}.json").exists()
                and json.loads((output / "results" / f"{entry['id']}.json").read_text()).get("status") == "exported"
            )
        )
        and (not args.match or any(term in entry["source"] for term in args.match))
    ]
    if args.limit:
        pending = pending[: args.limit]
    write_json(
        output / "run-plan.json",
        {
            "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "pending_models": len(pending),
            "chunk_size": args.chunk_size,
            "chunk_timeout_seconds": args.timeout,
            "output_policy": "Existing results never overwritten",
            "acceptance": "Nonempty USD with local dependencies",
            "machine": "Stallion; serial Kit workers on GPU 0",
            "monitor": "logs/, pipeline.pid, catalog.json",
        },
    )
    (output / "pipeline.pid").write_text(f"{os.getpid()}\n")
    try:
        while pending:
            if shutil.disk_usage(output).free < 40 * 1024**3:
                raise RuntimeError("Stopped with less than 40 GiB free disk")
            chunk, pending = pending[: args.chunk_size], pending[args.chunk_size :]
            log_path = output / "logs" / f"{time.time_ns()}.log"
            command = [
                "bash",
                str(ROOT / "tools/relling/python.sh"),
                "--no-ros-env",
                str(Path(__file__).resolve()),
                "worker",
                "--manifest",
                str(args.manifest.resolve()),
                "--ids",
                *[e["id"] for e in chunk],
            ]
            if args.audit:
                command.append("--audit")
            print(f"Converting {len(chunk)} models; log: {log_path}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT / "_build/linux-x86_64/release",
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=args.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = -1
                    if isinstance(error, KeyboardInterrupt):
                        raise
            missing = [entry for entry in chunk if not (records / f"{entry['id']}.json").exists()]
            if missing:
                first, *remaining = missing
                write_json(
                    records / f"{first['id']}.json",
                    {
                        "id": first["id"],
                        "source": first["source"],
                        "status": "failed",
                        "error": f"Worker exited {code} without a result; inspect {log_path}",
                        "log": str(log_path),
                    },
                )
                pending = remaining + pending
            summarize(manifest)
    finally:
        (output / "pipeline.pid").unlink(missing_ok=True)
        summarize(manifest)


def main() -> None:
    """Dispatch planning, supervised conversion, or a private worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--paths", type=Path, required=True)
    planner.add_argument("--source-base", type=Path, required=True)
    planner.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker", "status"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        if name in ("run", "worker"):
            command.add_argument("--audit", action="store_true")
        if name == "run":
            command.add_argument("--match", action="append")
            command.add_argument("--limit", type=int)
            command.add_argument("--chunk-size", type=int, default=10)
            command.add_argument("--timeout", type=int, default=600)
            command.add_argument("--retry-failed", action="store_true")
            command.add_argument("--retry-exported", action="store_true")
        elif name == "worker":
            command.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
    if args.action == "plan":
        plan(args)
    else:
        manifest = json.loads(args.manifest.read_text())
        if args.action == "run":
            run(args, manifest)
        elif args.action == "worker":
            worker(args, manifest)
        else:
            summarize(manifest)


if __name__ == "__main__":
    main()
