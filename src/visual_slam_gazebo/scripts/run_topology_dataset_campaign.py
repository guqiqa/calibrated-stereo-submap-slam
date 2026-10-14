#!/usr/bin/env python3
"""Run experiment_manager once per frozen topology layout."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest")
    parser.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    parser.add_argument("--methods", nargs="+", choices=("c0", "c1", "c4", "p"), default=["c1", "c4", "p"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=None,
                        help="override route-derived timeout for smoke tests")
    parser.add_argument("--query-routes", nargs="+", default=[
        "query_normal_01", "query_reverse_ring", "query_cross_hall",
        "query_cold_start", "query_none_of_the_above"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibration-model")
    parser.add_argument("--skip-mapping", action="store_true",
                        help="reuse an existing map database under each layout output")
    return parser.parse_args()


def route_timeout(path, speed_mps=0.2):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    distance = sum(math.hypot(float(right["x"]) - float(left["x"]),
                              float(right["y"]) - float(left["y"]))
                   for left, right in zip(rows, rows[1:]))
    return max(20.0, distance / speed_mps * 1.25 + 20.0)


def main():
    args = parse_args()
    payload = json.loads(Path(args.dataset_manifest).resolve().read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise SystemExit("dataset manifest is not passed")
    manager = Path(__file__).resolve().parent / "experiment_manager.py"
    jobs = []
    for layout in payload["layouts"]:
        if layout["split"] != args.split:
            continue
        roots = [("base", Path(layout["output"]))]
        if args.include_variants:
            roots.extend((mode, Path(item["output"])) for mode, item in layout["variants"].items())
        for variant, scene_root in roots:
            scene_root = scene_root.resolve()
            variant_output = (Path(args.output_root).resolve() /
                              f"{layout['family']}_{layout['seed']}" / variant)
            map_database = variant_output / "mapping" / "submap_database.yml"
            mapping_route = scene_root / "mapping_route.csv"
            if not args.skip_mapping:
                mapping_duration = args.duration or route_timeout(mapping_route)
                jobs.append([
                    sys.executable, str(manager), "--project", "topology_scene_smoke",
                    "--scene-manifest", str(scene_root / "scene_manifest.json"),
                    "--waypoint-route", str(mapping_route), "--methods", "c1", "--runs", "1",
                    "--duration", str(mapping_duration),
                    "--output-root", str(variant_output / "mapping"),
                    "--map-database", str(map_database), "--map-database-mode", "build",
                    "--shutdown-gazebo", "--gui" if args.gui else "--no-gui",
                ])
            for route_name in args.query_routes:
                route = scene_root / "query_routes" / f"{route_name}.csv"
                if not route.exists():
                    raise SystemExit(f"query route does not exist: {route}")
                output = variant_output / route_name
                duration = args.duration or route_timeout(route)
                command = [
                    sys.executable, str(manager), "--project", "topology_scene_smoke",
                    "--scene-manifest", str(scene_root / "scene_manifest.json"),
                    "--waypoint-route", str(route),
                    "--methods", *args.methods, "--runs", str(args.runs),
                    "--duration", str(duration), "--output-root", str(output),
                    "--map-database", str(map_database), "--map-database-mode", "load",
                    "--analyze", "--shutdown-gazebo",
                    "--gui" if args.gui else "--no-gui",
                ]
                if args.record:
                    command.append("--record")
                if args.archive:
                    command.append("--archive")
                if args.calibration_model:
                    command.extend(["--calibration-model", str(Path(args.calibration_model).resolve())])
                jobs.append(command)
    if not jobs:
        raise SystemExit(f"no layouts found for split {args.split}")
    if args.dry_run:
        print(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False))
        return
    for index, command in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
