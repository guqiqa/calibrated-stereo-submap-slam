#!/usr/bin/env python3
"""Write the delivered angular covariate for one run.

The paper condition is the DELIVERED rate, not the command. This reads the run's
sim_odom.csv (truth) and control_diagnostics.csv (phase labels and calibration
identity) and writes session_delivery.json next to them.

Usage: session_delivery_covariate.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_odom(path):
    series = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series.append((float(row["stamp"]),
                           yaw_from_quaternion(float(row["qx"]), float(row["qy"]),
                                               float(row["qz"]), float(row["qw"]))))
    return unwrap(series)


def unwrap(series):
    result = []
    offset = 0.0
    previous = None
    for stamp, yaw in series:
        if previous is not None:
            delta = yaw - previous
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            offset += delta - (yaw - previous)
        result.append((stamp, yaw + offset))
        previous = yaw
    return result


def slope(samples):
    n = float(len(samples))
    mt = sum(t for t, _ in samples) / n
    my = sum(y for _, y in samples) / n
    num = sum((t - mt) * (y - my) for t, y in samples)
    den = sum((t - mt) ** 2 for t, _ in samples)
    return num / den if den > 0 else float("nan")


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] if ordered else float("nan")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: session_delivery_covariate.py <run_dir>")
        return 2
    run_dir = Path(sys.argv[1])
    odom = load_odom(run_dir / "sim_odom.csv")

    rows = list(csv.DictReader((run_dir / "control_diagnostics.csv").open(
        newline="", encoding="utf-8")))
    blocks, active = [], []
    session_id = ""
    for row in rows:
        try:
            command = float(row.get("angular_z") or 0.0)
            stamp = float(row.get("stamp") or 0.0)
        except (TypeError, ValueError):
            continue
        if row.get("phase") == "rotate" and abs(command) > 1e-6:
            active.append({"stamp": stamp, "command": command})
            session_id = row.get("calibration_session_id") or session_id
        else:
            if len(active) >= 5:
                blocks.append(active)
            active = []
    if len(active) >= 5:
        blocks.append(active)

    windows = []
    for block in blocks:
        peak = max(abs(item["command"]) for item in block)
        plateau = [item for item in block if abs(item["command"]) >= 0.99 * peak]
        begin, end = plateau[0]["stamp"], plateau[-1]["stamp"]
        inner = [(t, y) for t, y in odom if begin + 0.2 <= t <= end - 0.2]
        if len(inner) < 10:
            continue
        rate = slope(inner)
        per = [abs((b[1] - a[1]) / (b[0] - a[0]))
               for a, b in zip(inner, inner[1:]) if b[0] - a[0] > 1e-9]
        windows.append({
            "start_sim_time": round(begin, 3),
            "end_sim_time": round(end, 3),
            "command": round(statistics.fmean([abs(item["command"]) for item in plateau]), 6),
            "delivered_angular_radps": round(abs(rate), 6),
            "delivered_p05": round(percentile(per, 0.05), 6),
            "delivered_p95": round(percentile(per, 0.95), 6),
            "ratio": round(abs(rate) / abs(plateau[0]["command"]), 6),
        })

    rates = [item["delivered_angular_radps"] for item in windows]
    payload = {
        "run_dir": str(run_dir),
        "source": "sim_odom",
        "windows": windows,
        "delivered_angular_radps": round(statistics.fmean(rates), 6) if rates else None,
        "delivered_angular_p05": round(percentile(rates, 0.05), 6) if rates else None,
        "delivered_angular_p95": round(percentile(rates, 0.95), 6) if rates else None,
        "calibration_session_id": session_id,
        "calibration_source": "session_v1" if session_id else "disabled_or_legacy",
        "calibration_valid": bool(session_id),
        "windows_measured": len(windows),
    }
    output = run_dir / "session_delivery.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
