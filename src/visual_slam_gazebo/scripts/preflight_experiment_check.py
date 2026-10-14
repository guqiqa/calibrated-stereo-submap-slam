#!/usr/bin/env python3
"""Produce conservative Gate A-D readiness reports before formal experiments."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess


def check(name, passed, detail, critical=True):
    return {"name": name, "status": "passed" if passed else "failed",
            "detail": detail, "critical": critical}


def status_rows(root):
    rows = []
    if not root:
        return rows
    for path in Path(root).rglob("status.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
            except json.JSONDecodeError:
                pass
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--status-root")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--calibration-model")
    parser.add_argument("--separation-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    source = (repo / "src/visual_slam/src/stereo_vo_sim.cpp").read_text(encoding="utf-8")
    rows = status_rows(args.status_root)
    methods = {row.get("retrieval_method") for row in rows}
    manifests = []
    if args.status_root:
        for path in Path(args.status_root).rglob("experiment_manifest.json"):
            try:
                manifests.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    scene_versions = {item.get("scene_map_version") for item in manifests if item.get("scene_map_version")}
    scene_schemas = {item.get("scene_schema_version") for item in manifests
                     if item.get("scene_schema_version") is not None}
    route_names = {Path(item["waypoint_route"]).stem for item in manifests if item.get("waypoint_route")}
    required_routes = {"query_same_view_inner", "query_same_view_outer",
                       "query_reverse_inner", "query_reverse_outer",
                       "query_cross_inner_outer", "query_cross_hall",
                       "query_cold_start", "query_none_of_the_above"}
    obstacle_versions = {item.get("scene_map_version") for item in manifests
                         if item.get("scene_obstacle_mode") in {"visual_only", "topology_change"}}
    separation_passed = False
    separation_detail = "no frame/route separation report supplied"
    if args.separation_report and Path(args.separation_report).exists():
        separation = json.loads(Path(args.separation_report).read_text(encoding="utf-8"))
        separation_passed = separation.get("status") == "passed"
        separation_detail = f"status={separation.get('status')}"
    query_manifests = [item for item in manifests
                       if item.get("waypoint_route") and Path(item["waypoint_route"]).stem != "mapping_route"]
    frozen_maps_loaded = bool(query_manifests) and all(
        item.get("map_database_mode") == "load" and item.get("map_database")
        for item in query_manifests)
    mapping_manifests = {item.get("map_database"): item for item in manifests
                         if item.get("map_database_mode") == "build" and item.get("map_database")}
    independent_captures = bool(query_manifests)
    capture_details = []
    for item in query_manifests:
        mapping = mapping_manifests.get(item.get("map_database"))
        separation_s = None
        valid = mapping is not None
        if valid:
            try:
                separation_s = (datetime.fromisoformat(item["started_at"]) -
                                datetime.fromisoformat(mapping["ended_at"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                valid = False
            valid = (valid and separation_s is not None and separation_s >= 5.0 and
                     mapping.get("scene_map_version") == item.get("scene_map_version") and
                     bool(item.get("map_database_sha256")))
        independent_captures = independent_captures and valid
        capture_details.append({"route": Path(item["waypoint_route"]).stem,
                                "separation_s": separation_s, "passed": valid})
    longest_failure_run = current = 0
    previous_source = None
    for row in rows:
        if row["_source_file"] != previous_source:
            current = 0
            previous_source = row["_source_file"]
        if row.get("state") in {"tracking_lost", "insufficient_stereo_features"}:
            current += 1
            longest_failure_run = max(longest_failure_run, current)
        else:
            current = 0
    eligible = [row for row in rows if row.get("evaluation_eligible", row.get("state") == "tracking")]
    evaluable = [row for row in eligible if row.get("coverage_label_available")]
    gate_a = [
        check("scene ID interface", all(token in source for token in
              ("submap_truth_path", "runtime_scene_mapping", "online_submap_id_mode")),
              "node exposes explicit runtime-to-scene ID alignment"),
        check("fixed candidate quotas", all(token in source for token in
              ("top_ids(visual_scores, visual_candidate_count_)",
               "top_ids(pose_scores, pose_candidate_count_)",
               "top_ids(topology_scores, topology_candidate_count_)",
               "merged_candidate_ids")),
              "Kv=4, Kp=3, Kg=2 and deterministic merge are present"),
        check("no online truth subscription", '"/sim/odom"' not in source,
              "visual node source must not subscribe to simulation truth"),
        check("heterogeneous submap schema", all(token in
              (repo / "src/visual_slam_gazebo/scripts/topology_scene_generator.py").read_text(encoding="utf-8")
              for token in ("hall_cell_size_m", "junction_radius_m", "numeric_id")),
              "corridor/hall/junction schema is implemented"),
        check("dedicated coverage calibration", all(token in source for token in
              ("candidate-coverage-v1", "coverage_model_", "last_coverage_features_")),
              "q_cov must use the frozen 12-feature model instead of q_top1 remapping"),
    ]
    gate_b = [
        check("C1/C4/P observed", {"c1", "c4", "p"}.issubset(methods),
              f"observed methods: {sorted(str(item) for item in methods)}"),
        check("six pilot layouts", len(scene_versions) >= 6,
              f"observed {len(scene_versions)} scene/map versions"),
        check("scene schema v2", bool(scene_schemas) and scene_schemas == {2},
              f"observed schema versions: {sorted(scene_schemas)}"),
        check("eight preregistered query routes", required_routes.issubset(route_names),
              f"observed routes: {sorted(route_names)}"),
        check("obstacle variant", bool(obstacle_versions),
              f"observed obstacle versions: {sorted(obstacle_versions)}"),
        check("independent frozen mapping", frozen_maps_loaded,
              f"query manifests using map_database_mode=load: "
              f"{sum(item.get('map_database_mode') == 'load' for item in query_manifests)}/"
              f"{len(query_manifests)}"),
        check("independent captures >=5 s", independent_captures,
              f"capture checks: {capture_details}"),
        check("mapping/query separation", separation_passed, separation_detail),
        check("tracking continuity", bool(rows) and longest_failure_run < 3,
              f"rows={len(rows)}, max consecutive failures={longest_failure_run}"),
    ]
    calibration_frozen = False
    calibration_detail = args.calibration_model or "no calibration model supplied"
    if args.calibration_model and Path(args.calibration_model).exists():
        calibration_payload = json.loads(Path(args.calibration_model).read_text(encoding="utf-8"))
        calibration_frozen = (calibration_payload.get("stage") == "confidence" and
                              calibration_payload.get("protocol_version") == "c1-c4-p-typed-query-v2" and
                              calibration_payload.get("coverage_feature_schema") ==
                              "candidate-coverage-v1" and
                              isinstance(calibration_payload.get("coverage_model"), dict) and
                              all(len(calibration_payload.get("coverage_model", {}).get(name, [])) == 12
                                  for name in ("weights", "mean", "scale")))
        calibration_detail = (f"version={calibration_payload.get('version')}, "
                              f"stage={calibration_payload.get('stage')}, "
                              f"protocol={calibration_payload.get('protocol_version')}, "
                              f"coverage_schema={calibration_payload.get('coverage_feature_schema')}")
    gate_c = [
        check("coverage labels complete", bool(evaluable) and len(evaluable) == len(eligible),
              f"{len(evaluable)}/{len(eligible)} evaluable rows labeled"),
        check("common ID space", bool(evaluable) and all(
              isinstance(row.get("candidate_ids"), list) and
              all(isinstance(value, str) for value in row.get("candidate_ids", [])) and
              all(isinstance(value, str) for value in row.get("legal_submaps", []))
              for row in evaluable), "candidate and legal IDs must both be scene strings"),
        check("calibration frozen", calibration_frozen, calibration_detail),
        check("q_top1/q_cov available", bool(evaluable) and all(
              row.get("q_top1") is not None and row.get("q_cov") is not None for row in evaluable),
              "both probabilities are required on every evaluable frame"),
        check("coverage features available", bool(evaluable) and all(
              isinstance(row.get("coverage_features"), list) and
              len(row.get("coverage_features")) == 12 for row in evaluable),
              "candidate-coverage-v1 requires 12 ordered features on every evaluable frame"),
    ]
    dataset_ok = False
    dataset_detail = "no full dataset manifest supplied"
    if args.dataset_manifest and Path(args.dataset_manifest).exists():
        payload = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
        dataset_ok = (payload.get("frozen") is True and
                      payload.get("protocol_version") == "c1-c4-p-typed-query-v2" and
                      bool(payload.get("test_layouts")))
        dataset_detail = (f"frozen={payload.get('frozen')}, "
                          f"protocol={payload.get('protocol_version')}, "
                          f"test_layouts={len(payload.get('test_layouts', []))}")
    gates = {"A_code_protocol": gate_a, "B_pilot_runtime": gate_b,
             "C_labels_calibration": gate_c,
             "D_formal_test_freeze": [check("full test dataset frozen", dataset_ok, dataset_detail)]}
    gate_status = {name: all(item["status"] == "passed" for item in items if item["critical"])
                   for name, items in gates.items()}
    separation_blocker = None
    if not separation_passed:
        separation_blocker = {
            "type": "typed_query_independence_not_satisfied",
            "thresholds_are_query_type_specific": True,
            "required": {"capture_time_s": 5.0, "reverse_opportunity_view_deg": 120.0,
                         "crossing_opportunity_view_deg": 30.0},
            "action": "fix_query_specific_independence_or_route_role_labels_before_calibration_test",
            "forbidden": [
                "delete_failed_frames",
                "apply_space_or_view_thresholds_to_all_same_view_frames",
                "reuse_mapping_frames_or_feature_cache",
            ],
        }
    report = {"formal_test_ready": all(gate_status.values()), "gate_status": gate_status,
              "gates": gates,
              "separation_blocker": separation_blocker,
              "note": "mapping/query leakage prevention is global; spatial and view checks are query-type-specific and never delete frames"}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Formal Experiment Preflight", "", f"Formal test ready: **{report['formal_test_ready']}**", ""]
    for name, items in gates.items():
        lines.extend([f"## {name}", ""])
        lines.extend(f"- [{'x' if item['status'] == 'passed' else ' '}] {item['name']}: {item['detail']}"
                     for item in items)
        lines.append("")
    lines.append(report["note"])
    (output / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["formal_test_ready"] else 2)


if __name__ == "__main__":
    main()
