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


def export_one(entry: dict) -> dict:
    """Convert one source without overrides and verify a nonempty, portable USD stage."""
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    source = Path(entry["source"])
    if digest(source) != entry["sha256"]:
        raise RuntimeError("Source XML changed after inventory")
    destination = Path(entry["output"])
    destination.mkdir(parents=True, exist_ok=False)
    importer = MJCFImporter(
        MJCFImporterConfig(
            mjcf_path=str(source),
            usd_path=str(destination),
            import_scene=True,
            fix_base=None,
            allow_self_collision=True,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
    )
    usd_path = Path(importer.import_mjcf()).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError("Importer output cannot be opened")
    prims = list(stage.Traverse())
    geometry = sum(prim.IsA(UsdGeom.Gprim) for prim in prims)
    if not geometry:
        raise RuntimeError("Imported stage contains no geometry")
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(usd_path)))
    dependencies = [layer.realPath for layer in layers if layer.realPath] + list(assets)
    external = [str(path) for path in dependencies if path and not Path(path).resolve().is_relative_to(destination)]
    if unresolved or external:
        raise RuntimeError(f"USD dependencies are incomplete or external: unresolved={unresolved}, external={external}")
    if digest(source) != entry["sha256"]:
        raise RuntimeError("Source XML changed during export")
    result = {
        "status": "exported",
        "usd_path": str(usd_path),
        "prim_count": len(prims),
        "geometry_count": geometry,
        "rigid_body_count": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims),
        "joint_count": sum(prim.IsA(UsdPhysics.Joint) for prim in prims),
        "dependency_count": len(dependencies),
        "source_sha256": entry["sha256"],
        "physics_equivalence": "not_tested",
        "controller_port": "not_performed",
    }
    del prims, stage, layers, importer
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
                result.update(export_one(entry))
            except Exception as error:  # noqa: BLE001 - isolate failures at the vendor model boundary
                result["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            result["seconds"] = round(time.monotonic() - start, 3)
            write_json(Path(manifest["output_root"]) / "results" / f"{entry['id']}.json", result)
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
        entries.append({**entry, **result})
    counts = dict(collections.Counter(entry["status"] for entry in entries))
    summary = {"output_root": str(output), "counts": counts, "limitations": manifest["limitations"], "entries": entries}
    write_json(output / "catalog.json", summary)
    print(json.dumps({"output_root": str(output), "counts": counts}), flush=True)
    return summary


def run(args: argparse.Namespace, manifest: dict) -> None:
    """Supervise serial chunks, retaining progress across a crash or deadline."""
    output = Path(manifest["output_root"])
    (output / "results").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    pending = [
        entry
        for entry in manifest["entries"]
        if entry["kind"] == "model"
        and not (output / "results" / f"{entry['id']}.json").exists()
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
            missing = [entry for entry in chunk if not (output / "results" / f"{entry['id']}.json").exists()]
            if missing:
                first, *remaining = missing
                write_json(
                    output / "results" / f"{first['id']}.json",
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
        if name == "run":
            command.add_argument("--match", action="append")
            command.add_argument("--limit", type=int)
            command.add_argument("--chunk-size", type=int, default=10)
            command.add_argument("--timeout", type=int, default=600)
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
