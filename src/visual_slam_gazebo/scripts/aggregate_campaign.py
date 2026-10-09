#!/usr/bin/env python3
"""Aggregate repeated G/S/P campaign results without changing raw runs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


MODES = ("global", "fixed", "joint")


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def numeric_status(run_dir, field):
    path = run_dir / "status.jsonl"
    values = []
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line).get(field)
            if value is not None:
                values.append(float(value))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return values


def run_record(mode, run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = {}
    for field in ("matching_ms", "processing_ms", "active_map_points", "valid_matches",
                  "median_disparity_px", "stereo_reprojection_px"):
        values = numeric_status(run_dir, field)
        metrics[f"mean_{field}"] = mean(values) if values else None
        metrics[f"p95_{field}"] = percentile(values, 0.95) if values else None
        metrics[f"std_{field}"] = stdev(values) if len(values) > 1 else 0.0 if values else None
    record = {
        "mode": mode,
        "run": int(run_dir.name.split("_")[-1]),
        "path_ratio": summary.get("visual_truth_distance_ratio"),
        "truth_path_m": summary.get("truth_path_distance_m"),
        "visual_path_m": summary.get("visual_path_distance_m"),
        "closure_error_m": summary.get("visual_closure_error_m"),
        "tracking_failures": summary.get("failure_messages", 0),
        "max_consecutive_tracking_failures": summary.get("max_consecutive_tracking_failures", 0),
        "switch_count": summary.get("switch_count", 0),
        "wrong_switch_count": summary.get("wrong_switch_count", 0),
        "loop_closure_events": summary.get("loop_closure_events", 0),
    }
    record.update(metrics)
    record["pass"] = (
        0.98 <= float(record["path_ratio"] or 0.0) <= 1.02
        and int(record["tracking_failures"] or 0) == 0
        and int(record["max_consecutive_tracking_failures"] or 0) == 0
        and int(record["wrong_switch_count"] or 0) == 0
    )
    return record


def summarize(records):
    output = {}
    fields = ("path_ratio", "closure_error_m", "mean_matching_ms",
              "mean_processing_ms", "mean_active_map_points", "switch_count",
              "wrong_switch_count")
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        mode_summary = {"runs": len(rows), "passed": sum(row["pass"] for row in rows)}
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            mode_summary[field] = {
                "mean": mean(values) if values else None,
                "std": stdev(values) if len(values) > 1 else 0.0 if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        output[mode] = mode_summary
    return output


def write_plots(output, records, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    labels = list(MODES)
    colors = ["#4c78a8", "#59a14f", "#f28e2b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for axis, field, title, ylabel in zip(
        axes,
        ("path_ratio", "mean_matching_ms", "mean_active_map_points"),
        ("Scale", "Matching time", "Active map points"),
        ("Visual / truth path", "Mean ms", "Mean points"),
    ):
        means = [summary[mode][field]["mean"] for mode in labels]
        errors = [summary[mode][field]["std"] or 0.0 for mode in labels]
        axis.bar(labels, means, yerr=errors, color=colors, capsize=4)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("45 m round-trip G/S/P campaign")
    fig.savefig(output / "campaign_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for mode, color in zip(labels, colors):
        rows = [row for row in records if row["mode"] == mode]
        axes[0].plot([row["run"] for row in rows], [row["path_ratio"] for row in rows], "o-", label=mode, color=color)
        axes[1].plot([row["run"] for row in rows], [row["mean_processing_ms"] for row in rows], "o-", label=mode, color=color)
    axes[0].axhline(1.0, color="#555", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Visual / truth path")
    axes[1].set_ylabel("Mean processing time (ms)")
    for axis in axes:
        axis.set_xlabel("Run")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Repeatability across 10 runs")
    fig.savefig(output / "campaign_repeatability.png", dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = (args.output or campaign / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in MODES:
        for run_dir in sorted((campaign / mode).glob("run_*")):
            record = run_record(mode, run_dir)
            if record is not None:
                records.append(record)
    records.sort(key=lambda row: (row["mode"], row["run"]))
    fields = list(records[0]) if records else []
    with (output / "campaign_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    aggregate = {
        "campaign": str(campaign),
        "total_runs": len(records),
        "complete_expected_runs": len(records) == 30,
        "pass_runs": sum(row["pass"] for row in records),
        "by_mode": summarize(records),
        "plots_generated": write_plots(output, records, summarize(records)),
    }
    (output / "campaign_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
