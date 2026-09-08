#!/usr/bin/env python3
"""Summarize SLAM status metrics in fixed ground-truth route segments."""

import argparse
import bisect
import csv
import json
from pathlib import Path
import statistics


FAILURE_STATES = {"tracking_lost", "insufficient_stereo_features"}


def mean(values):
    return statistics.fmean(values) if values else None


def load_truth(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "host_time" not in rows[0]:
        raise ValueError("sim_odom.csv requires host_time; rerun with the updated recorder")
    truth = []
    for row in rows:
        try:
            truth.append((float(row["host_time"]), float(row["x"]), float(row["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not truth:
        raise ValueError("sim_odom.csv contains no valid truth samples")
    truth.sort()
    return truth


def nearest_position(truth, timestamps, host_time):
    index = bisect.bisect_left(timestamps, host_time)
    candidates = []
    if index < len(truth):
        candidates.append(truth[index])
    if index > 0:
        candidates.append(truth[index - 1])
    return min(candidates, key=lambda item: abs(item[0] - host_time))


def load_statuses(path, truth):
    timestamps = [item[0] for item in truth]
    statuses = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
                host_time = float(data["host_time"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            _, x, y = nearest_position(truth, timestamps, host_time)
            data["truth_x"] = x
            data["truth_y"] = y
            statuses.append(data)
    return statuses


def make_rows(statuses, segment_size, max_position):
    buckets = {}
    for item in statuses:
        position = max(0.0, item["truth_x"])
        if position > max_position:
            continue
        segment = int(position // segment_size)
        buckets.setdefault(segment, []).append(item)

    rows = []
    for segment in sorted(buckets):
        items = buckets[segment]
        states = [item.get("state") for item in items]
        tracking = sum(state == "tracking" for state in states)
        failures = sum(state in FAILURE_STATES for state in states)
        switches = [int(item.get("switch_count", 0)) for item in items]
        wrong_switches = [int(item.get("wrong_switch_count", 0)) for item in items]
        rows.append({
            "segment_start_m": segment * segment_size,
            "segment_end_m": (segment + 1) * segment_size,
            "status_messages": len(items),
            "tracking_messages": tracking,
            "failure_messages": failures,
            "tracking_rate": tracking / len(items),
            "failure_rate": failures / len(items),
            "mean_active_map_points": mean([float(item.get("active_map_points", 0)) for item in items]),
            "mean_matching_ms": mean([float(item.get("matching_ms", 0.0)) for item in items]),
            "mean_processing_ms": mean([float(item.get("processing_ms", 0.0)) for item in items]),
            "switches_in_segment": max(switches) - min(switches) if switches else 0,
            "wrong_switches_in_segment": max(wrong_switches) - min(wrong_switches) if wrong_switches else 0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--segment-size", type=float, default=5.0)
    parser.add_argument("--max-position", type=float, default=40.0)
    args = parser.parse_args()
    if args.segment_size <= 0.0 or args.max_position <= 0.0:
        raise SystemExit("segment-size and max-position must be positive")

    truth = load_truth(args.run_dir / "sim_odom.csv")
    statuses = load_statuses(args.run_dir / "status.jsonl", truth)
    rows = make_rows(statuses, args.segment_size, args.max_position)
    csv_path = args.run_dir / "segment_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "segment_size_m": args.segment_size,
        "max_position_m": args.max_position,
        "source_status_messages": len(statuses),
        "segment_count": len(rows),
        "segments": rows,
    }
    (args.run_dir / "segment_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
