#!/usr/bin/env python3
"""Generate parameterized hall/double-ring/radial Gazebo scenes."""

import argparse
import json
from pathlib import Path
import sys

from topology_scene_generator import FAMILIES, OBSTACLE_MODES, generate_scene, load_config


def repository_root():
    return Path(__file__).resolve().parents[3]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="regular_baseline")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--config", default=str(repository_root() / "config" / "topology_scene_defaults.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--split", choices=("development", "calibration", "test"))
    parser.add_argument("--obstacle-mode", choices=OBSTACLE_MODES, default="none")
    parser.add_argument("--list-families", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--generate-dataset", action="store_true")
    parser.add_argument("--dataset-profile", choices=("pilot", "full"), default="pilot",
                        help="Dataset size: pilot generates 6 layouts; full generates the formal set")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export-rmf", action="store_true")
    parser.add_argument("--export-gazebo", action="store_true")
    return parser.parse_args()


def dataset_split(index, counts):
    if index < counts["development"]:
        return "development"
    if index < counts["development"] + counts["calibration"]:
        return "calibration"
    return "test"


def main():
    args = parse_args()
    config = load_config(Path(args.config))
    if args.list_families:
        print("\n".join(FAMILIES))
        return
    if args.show_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    output_root = Path(args.output or repository_root() / "generated_scenes")
    export_rmf = args.export_rmf or not (args.export_rmf or args.export_gazebo)
    export_gazebo = args.export_gazebo or not (args.export_rmf or args.export_gazebo)
    preview = args.preview or args.generate_dataset
    if args.generate_dataset:
        datasets = config.get("dataset_profiles", {})
        dataset = datasets.get(args.dataset_profile, config["dataset"])
        results = []
        for family in dataset["families"]:
            for index in range(int(dataset["per_family"])):
                split = dataset_split(index, dataset["split_counts"])
                seed = int(dataset["seed_bases"][family]) + index
                destination = output_root / split / f"{family}_{seed}"
                result = generate_scene(args.config, family, seed, "none", split, destination,
                                        export_rmf, export_gazebo, preview)
                variants = {}
                for mode in ("visual_only", "topology_change"):
                    variant_output = destination / "variants" / mode
                    variant = generate_scene(args.config, family, seed, mode, split, variant_output,
                                             export_rmf, export_gazebo, False)
                    variants[mode] = {"output": str(variant_output), "status": variant["status"]}
                results.append({"family": family, "seed": seed, "split": split,
                                "output": str(destination), "status": result["status"],
                                "variants": variants})
        manifest = {"status": "passed" if all(
                        row["status"] == "passed" and
                        all(item["status"] == "passed" for item in row["variants"].values())
                        for row in results) else "failed",
                    "dataset_profile": args.dataset_profile,
                    "layout_count": len(results), "layouts": results}
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False))
        return
    destination = output_root if args.output else output_root / f"{args.family}_{args.seed}_{args.obstacle_mode}"
    result = generate_scene(args.config, args.family, args.seed, args.obstacle_mode, args.split,
                            destination, export_rmf, export_gazebo, preview or args.validate)
    print(json.dumps({"output": str(destination), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
