#!/usr/bin/env python3
"""Run data-quality, retrieval, route, campaign, and scaling analyses."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


METHODS = ("c0", "c1", "c4", "p", "global", "fixed", "joint")


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def load_status(path):
    rows, invalid = [], 0
    if not path.exists():
        return rows, invalid
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid += 1
    return rows, invalid


def max_failure_run(rows):
    longest = current = 0
    for row in rows:
        if row.get("state") in {"tracking_lost", "insufficient_stereo_features"}:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def write_event_tables(run_dir, rows):
    candidate_fields = ["host_time", "retrieval_method", "active_submap", "candidate_ids",
                        "candidate_covered", "coverage_label_available", "q_top1", "q_cov"]
    geometry_fields = ["host_time", "retrieval_method", "selected_submap", "geometry_verified",
                       "pnp_inliers", "reprojection_error_px", "retrieval_decision"]
    rejection_fields = ["host_time", "retrieval_method", "retrieval_decision", "reject_reason"]
    for name, fields in (("candidate_labels.csv", candidate_fields),
                         ("geometry_verification.csv", geometry_fields),
                         ("reject_reasons.csv", rejection_fields)):
        with (run_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                output = {key: row.get(key) for key in fields}
                if isinstance(output.get("candidate_ids"), list):
                    output["candidate_ids"] = json.dumps(output["candidate_ids"])
                writer.writerow(output)


def run_quality(run_dir):
    required = ["status.jsonl", "visual_odom.csv", "sim_odom.csv", "summary.json",
                "frame_diagnostics.csv", "resolved_parameters.yaml", "stereo_vo_sim.log"]
    rows, invalid = load_status(run_dir / "status.jsonl")
    files = {name: {"exists": (run_dir / name).exists(),
                    "nonempty": (run_dir / name).exists() and (run_dir / name).stat().st_size > 0}
             for name in required}
    states = [row.get("state") for row in rows]
    report = {
        "run": str(run_dir),
        "run_state": "complete",
        "files": files,
        "status_messages": len(rows),
        "invalid_status_lines": invalid,
        "max_consecutive_tracking_failures": max_failure_run(rows),
        "retrieval_fields_available": all(any(key in row for row in rows) for key in
                                           ("retrieval_method", "geometry_verified", "reject_reason")),
        "coverage_labels_available": any(row.get("coverage_label_available") for row in rows),
        "stereo_image_continuity": "available_in_rosbag" if (run_dir / "rosbag").exists() else "unavailable_no_rosbag",
    }
    if not rows or invalid or any(not value["nonempty"] for value in files.values()):
        report["run_state"] = "incomplete"
    if report["max_consecutive_tracking_failures"] >= 3:
        report["run_state"] = "incomplete"
    (run_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_event_tables(run_dir, rows)
    return rows, report


def runtime_record(run_dir, rows):
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    def numbers(field):
        return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    processing, matching, points = numbers("processing_ms"), numbers("matching_ms"), numbers("active_map_points")
    resource_rows = []
    resource_path = run_dir / "resource_usage.csv"
    if resource_path.exists():
        with resource_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    resource_rows.append((float(row["rss_kb"]), float(row["cpu_percent"])))
                except (KeyError, TypeError, ValueError):
                    continue
    route_m = None
    for parent in run_dir.parents:
        if parent.name.endswith("m") and parent.name[:-1].isdigit():
            route_m = int(parent.name[:-1])
            break
    return {
        "method": run_dir.parent.name,
        "run": run_dir.name,
        "route_m": route_m,
        "state": "complete" if rows else "incomplete",
        "mean_processing_ms": statistics.fmean(processing) if processing else None,
        "p95_processing_ms": percentile(processing, .95),
        "mean_matching_ms": statistics.fmean(matching) if matching else None,
        "p95_matching_ms": percentile(matching, .95),
        "mean_active_map_points": statistics.fmean(points) if points else None,
        "peak_rss_mb": max((row[0] for row in resource_rows), default=0.0) / 1024.0 if resource_rows else None,
        "mean_cpu_percent": statistics.fmean(row[1] for row in resource_rows) if resource_rows else None,
        "tracking_failures": summary.get("failure_messages"),
        "path_ratio": summary.get("visual_truth_distance_ratio"),
        "switch_count": summary.get("switch_count"),
        "wrong_switch_count": summary.get("wrong_switch_count"),
    }


def optional_plots(output, records):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    usable = [row for row in records if row["mean_processing_ms"] is not None]
    if not usable:
        return False
    methods = sorted(set(row["method"] for row in usable))
    means = [statistics.fmean(row["mean_processing_ms"] for row in usable if row["method"] == method)
             for method in methods]
    points = [statistics.fmean(row["mean_active_map_points"] for row in usable
                              if row["method"] == method and row["mean_active_map_points"] is not None)
              for method in methods]
    for name, values, ylabel in (("method_comparison.png", means, "Mean processing time (ms)"),
                                 ("processing_time.png", means, "Mean processing time (ms)"),
                                 ("active_map_points.png", points, "Mean active map points")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(methods, values)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
    return True


def write_protocol_aggregates(output, all_rows):
    methods = sorted(set(row.get("retrieval_method", "unknown") for row in all_rows))
    coverage_rows, geometry_rows, calibration_rows = [], [], []
    reject_counts = {}
    for method in methods:
        rows = [row for row in all_rows if row.get("retrieval_method", "unknown") == method]
        labeled = [row for row in rows if row.get("coverage_label_available") and
                   row.get("candidate_covered") is not None]
        covered = sum(bool(row.get("candidate_covered")) for row in labeled)
        verified = [row for row in rows if row.get("geometry_verification_available", True)]
        geometry_ok = sum(bool(row.get("geometry_verified")) for row in verified)
        coverage_rows.append({"method": method, "labeled_events": len(labeled),
                              "covered_events": covered,
                              "candidate_coverage": covered / len(labeled) if labeled else None,
                              "status": "available" if labeled else "unavailable"})
        geometry_rows.append({"method": method, "events": len(verified), "verified": geometry_ok,
                              "verification_rate": geometry_ok / len(verified) if verified else None})
        calibration_rows.append({"method": method, "status": "unavailable" if not labeled else "pending_evaluator",
                                 "reason": "missing independent mapping/query labels" if not labeled else ""})
        for row in rows:
            reason = row.get("reject_reason") or "none"
            reject_counts[(method, reason)] = reject_counts.get((method, reason), 0) + 1
    tables = {
        "candidate_coverage.csv": coverage_rows,
        "geometry_verification.csv": geometry_rows,
        "calibration_by_method.csv": calibration_rows,
        "reject_reason_counts.csv": [
            {"method": method, "reject_reason": reason, "count": count}
            for (method, reason), count in sorted(reject_counts.items())],
    }
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else ["method"]
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    # The empty curve is intentional until independent labels exist.
    (output / "risk_coverage.csv").write_text("method,coverage,risk,count\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--route-m", type=float, default=45.0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    records, quality, all_rows = [], [], []
    for run_dir in sorted(path for path in root.rglob("run_*") if path.is_dir()):
        rows, report = run_quality(run_dir)
        all_rows.extend(rows)
        quality.append(report)
        records.append(runtime_record(run_dir, rows))
        if (run_dir / "sim_odom.csv").exists() and (run_dir / "status.jsonl").exists():
            subprocess.run([sys.executable, str(script_dir / "analyze_route_segments.py"), str(run_dir),
                            "--segment-size", "5.0", "--max-position", str(args.route_m)], check=False)
        subprocess.run([sys.executable, str(script_dir / "evaluate_retrieval_protocol.py"),
                        "--status-root", str(run_dir), "--output", str(run_dir / "retrieval_analysis")], check=False)
    fields = list(records[0]) if records else ["method", "run", "state"]
    with (output / "method_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {"runs": len(records), "complete": sum(item["run_state"] == "complete" for item in quality),
               "incomplete": sum(item["run_state"] != "complete" for item in quality),
               "coverage_metrics": "available" if any(item["coverage_labels_available"] for item in quality)
               else "unavailable: missing independent mapping/query labels",
               "plots_generated": optional_plots(output, records)}
    write_protocol_aggregates(output, all_rows)
    with (output / "calibration_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "status", "reason"])
        writer.writeheader()
        writer.writerow({"condition": "normal", "status": "unavailable",
                         "reason": "missing independent mapping/query labels"})
    scaling = [row for row in records if row.get("route_m") is not None]
    with (output / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scaling)
    (output / "scaling_summary.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")
    (output / "runtime_summary.json").write_text(json.dumps({"summary": summary, "runs": records}, indent=2), encoding="utf-8")
    (output / "data_quality_report.json").write_text(json.dumps({"summary": summary, "runs": quality}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
