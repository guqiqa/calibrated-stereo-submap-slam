#!/usr/bin/env python3
"""Audit mapping/query independence with query-type-specific conditions.

Every query frame is retained. Distance and viewpoint difference are saved as
diagnostics and become gates only for preregistered retrieval opportunities.
"""

import argparse
import csv
import json
import math
from pathlib import Path


QUERY_POLICIES = {
    "mapped_same_view": {"legal": "nonempty"},
    "mapped_reverse_view": {"legal": "nonempty", "opportunity_view_deg": 120.0},
    "mapped_crossing": {"legal": "nonempty", "opportunity_view_deg": 30.0},
    "cold_start": {"legal": "nonempty", "required_role": "relocalization_frame"},
    "none_of_the_above": {"legal": "empty", "required_role": "ood_frame"},
}


def parse_legal(value):
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{**row, "x": float(row["x"]), "y": float(row["y"]),
                 "yaw": float(row["yaw"]), "legal_submaps": parse_legal(row.get("legal_submaps"))}
                for row in csv.DictReader(handle)]


def angle_difference(left, right):
    return abs(math.atan2(math.sin(left - right), math.cos(left - right))) * 180.0 / math.pi


def nearest_mapping_pose(query, mapping):
    distances = [(math.hypot(query["x"] - item["x"], query["y"] - item["y"]), item)
                 for item in mapping]
    minimum = min(distance for distance, _ in distances)
    nearby = [item for distance, item in distances if distance <= minimum + 1e-6]
    nearest = min(nearby, key=lambda item: angle_difference(query["yaw"], item["yaw"]))
    return minimum, nearest, angle_difference(query["yaw"], nearest["yaw"])


def audit(mapping, query, capture_time_separation_s, min_time_s=5.0,
          reverse_view_deg=120.0, crossing_view_deg=30.0):
    query_types = {row.get("query_type", "") for row in query}
    route_type = next(iter(query_types)) if len(query_types) == 1 else "mixed_or_missing"
    policy = dict(QUERY_POLICIES.get(route_type, {}))
    if route_type == "mapped_reverse_view":
        policy["opportunity_view_deg"] = reverse_view_deg
    elif route_type == "mapped_crossing":
        policy["opportunity_view_deg"] = crossing_view_deg
    route_failures = []
    if not query:
        route_failures.append("empty_query_route")
    if not policy:
        route_failures.append("unknown_or_mixed_query_type")
    if capture_time_separation_s < min_time_s:
        route_failures.append("capture_time_separation_below_minimum")

    frame_reports = []
    opportunity_count = 0
    for row in query:
        distance, nearest, view = nearest_mapping_pose(row, mapping)
        role = row.get("frame_role", "")
        legal = row["legal_submaps"]
        reasons = []
        if policy.get("legal") == "nonempty" and not legal:
            reasons.append("normal_query_has_no_legal_submap")
        if policy.get("legal") == "empty" and legal:
            reasons.append("ood_query_has_legal_submap")
        if policy.get("required_role") and role != policy["required_role"]:
            reasons.append("invalid_frame_role")
        if role == "retrieval_opportunity_frame":
            opportunity_count += 1
            threshold = policy.get("opportunity_view_deg")
            if threshold is not None and view < threshold:
                reasons.append("retrieval_opportunity_view_below_threshold")
        frame_reports.append({
            "waypoint": int(row["waypoint"]), "query_type": row.get("query_type"),
            "frame_role": role, "nearest_mapping_waypoint": int(nearest["waypoint"]),
            "nearest_mapping_distance_m": distance, "nearest_mapping_view_deg": view,
            "legal_submaps": legal, "passed": not reasons, "failure_reasons": reasons,
        })
    if policy.get("opportunity_view_deg") is not None and opportunity_count == 0:
        route_failures.append("no_retrieval_opportunity_frames")
    failed_frames = [item for item in frame_reports if not item["passed"]]
    if failed_frames:
        route_failures.append("frame_policy_failure")
    return {
        "query_type": route_type, "policy": policy, "waypoints": len(query),
        "retrieval_opportunity_frames": opportunity_count,
        "failed_frames": len(failed_frames), "route_failure_reasons": route_failures,
        "status": "passed" if not route_failures else "failed",
        "frame_diagnostics": frame_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-route", required=True)
    parser.add_argument("--query-routes", nargs="+", required=True)
    parser.add_argument("--capture-time-separation-s", type=float, required=True)
    parser.add_argument("--min-time-s", type=float, default=5.0)
    parser.add_argument("--reverse-view-deg", type=float, default=120.0)
    parser.add_argument("--crossing-view-deg", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mapping_path = Path(args.mapping_route).resolve()
    mapping = rows(mapping_path)
    reports = []
    for route_value in args.query_routes:
        route_path = Path(route_value).resolve()
        report = audit(mapping, rows(route_path), args.capture_time_separation_s,
                       args.min_time_s, args.reverse_view_deg, args.crossing_view_deg)
        report["route"] = str(route_path)
        report["separate_route_file"] = route_path != mapping_path
        if not report["separate_route_file"]:
            report["status"] = "failed"
            report["route_failure_reasons"].append("mapping_route_reused_as_query")
        reports.append(report)
    passed = bool(reports) and all(item["status"] == "passed" for item in reports)
    payload = {
        "status": "passed" if passed else "failed",
        "protocol": "typed-query-independence-v2",
        "global_requirements": {
            "capture_time_separation_s": args.min_time_s, "separate_capture": True,
            "no_frame_timestamp_or_feature_cache_reuse": True,
            "query_map_database_mode": "load",
        },
        "capture_time_separation_s": args.capture_time_separation_s,
        "query_type_policies": QUERY_POLICIES, "routes": reports,
        "frame_filtering_applied": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "protocol": payload["protocol"],
                      "routes": [{key: item[key] for key in
                                  ("route", "query_type", "status", "waypoints",
                                   "retrieval_opportunity_frames", "failed_frames")}
                                 for item in reports]}, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
