#!/usr/bin/env python3
"""Audit conjunctive mapping/query separation without altering routes.

This checker deliberately does not repair routes or discard failed frames. A
failed report is a protocol blocker until the route is redesigned or the
preregistered protocol is formally revised.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{**row, "x": float(row["x"]), "y": float(row["y"]),
                 "yaw": float(row["yaw"])} for row in csv.DictReader(handle)]


def angle_difference(left, right):
    return abs(math.atan2(math.sin(left - right), math.cos(left - right))) * 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-route", required=True)
    parser.add_argument("--query-routes", nargs="+", required=True)
    parser.add_argument("--capture-time-separation-s", type=float, required=True)
    parser.add_argument("--min-time-s", type=float, default=5.0)
    parser.add_argument("--min-space-m", type=float, default=2.0)
    parser.add_argument("--min-view-deg", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mapping = rows(args.mapping_route)
    reports = []
    for route_path in args.query_routes:
        failures = []
        query = rows(route_path)
        for row in query:
            nearest = min(mapping, key=lambda item: math.hypot(row["x"] - item["x"], row["y"] - item["y"]))
            distance = math.hypot(row["x"] - nearest["x"], row["y"] - nearest["y"])
            view = angle_difference(row["yaw"], nearest["yaw"])
            if (args.capture_time_separation_s < args.min_time_s or
                    distance < args.min_space_m or view < args.min_view_deg):
                failures.append({"waypoint": int(row["waypoint"]), "nearest_mapping_waypoint": int(nearest["waypoint"]),
                                 "space_m": distance, "view_deg": view})
        reports.append({"route": str(Path(route_path).resolve()), "waypoints": len(query),
                        "failed": len(failures), "examples": failures[:20]})
    passed = args.capture_time_separation_s >= args.min_time_s and all(not item["failed"] for item in reports)
    report = {"status": "passed" if passed else "failed",
              "thresholds": {"time_s": args.min_time_s, "space_m": args.min_space_m,
                             "view_deg": args.min_view_deg},
              "capture_time_separation_s": args.capture_time_separation_s,
              "routes": reports}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
