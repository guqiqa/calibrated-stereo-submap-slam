#!/usr/bin/env python3
"""Run experiment_manager once per frozen topology layout."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest")
    parser.add_argument("--split", choices=("development", "calibration", "test"), required=True)
    parser.add_argument("--methods", nargs="+", choices=("c0", "c1", "c4", "p"), default=["c1", "c4", "p"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--query-route", default="query_routes/query_normal_01.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--include-variants", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
            output = Path(args.output_root).resolve() / f"{layout['family']}_{layout['seed']}" / variant
            command = [
                sys.executable, str(manager), "--project", "topology_scene_smoke",
                "--scene-manifest", str(scene_root / "scene_manifest.json"),
                "--waypoint-route", str(scene_root / args.query_route),
                "--methods", *args.methods, "--runs", str(args.runs),
                "--duration", str(args.duration), "--output-root", str(output),
                "--analyze", "--shutdown-gazebo",
                "--gui" if args.gui else "--no-gui",
            ]
            if args.record:
                command.append("--record")
            if args.archive:
                command.append("--archive")
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
