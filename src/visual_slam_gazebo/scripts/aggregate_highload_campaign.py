#!/usr/bin/env python3
"""Combine independently aggregated 60/90/120 m campaigns."""

import argparse
import csv
import json
from pathlib import Path


MODES = ("global", "fixed", "joint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing 60m, 90m and 120m")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for distance in (60, 90, 120):
        summary_path = root / f"{distance}m" / "analysis" / "campaign_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for mode in MODES:
            item = summary["by_mode"][mode]
            rows.append({
                "distance_m": distance,
                "mode": mode,
                "runs": item["runs"],
                "passed": item["passed"],
                "pass_rate": item["passed"] / item["runs"],
                "path_ratio_mean": item["path_ratio"]["mean"],
                "path_ratio_std": item["path_ratio"]["std"],
                "matching_ms_mean": item["mean_matching_ms"]["mean"],
                "processing_ms_mean": item["mean_processing_ms"]["mean"],
                "active_map_points_mean": item["mean_active_map_points"]["mean"],
            })

    fields = list(rows[0])
    with (output / "highload_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    combined = {"root": str(root), "distances_m": [60, 90, 120], "rows": rows}
    (output / "highload_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"global": "#4c78a8", "fixed": "#59a14f", "joint": "#f28e2b"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        x = [row["distance_m"] for row in selected]
        axes[0].plot(x, [row["path_ratio_mean"] for row in selected], "o-", label=mode, color=colors[mode])
        axes[1].plot(x, [row["matching_ms_mean"] for row in selected], "o-", label=mode, color=colors[mode])
        axes[2].plot(x, [row["active_map_points_mean"] for row in selected], "o-", label=mode, color=colors[mode])
    axes[0].axhline(0.98, color="#888", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Visual / truth path")
    axes[1].set_ylabel("Mean matching time (ms)")
    axes[2].set_ylabel("Mean active map points")
    for axis in axes:
        axis.set_xlabel("Route length (m)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("High-load route-length comparison")
    fig.savefig(output / "highload_scaling.png", dpi=180)
    plt.close(fig)

    print(json.dumps(combined, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
