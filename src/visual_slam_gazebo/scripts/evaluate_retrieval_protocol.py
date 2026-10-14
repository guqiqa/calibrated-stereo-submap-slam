#!/usr/bin/env python3
"""Replay retrieval diagnostics and compute calibrated metrics.

Legacy status logs do not contain a legal-submap label. Such logs are retained
as evidence replay, while coverage and calibration metrics remain unavailable
until a mapping/query protocol supplies labels.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from calibration_metrics import brier_score, ece, nll, risk_coverage_curve


def iter_status(root):
    for path in sorted(Path(root).rglob("status.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            item["_source_file"] = str(path)
            yield item


def method_for(item):
    return {"global": "c1", "fixed": "c1", "joint": "c4"}.get(
        item.get("submap_mode"), item.get("retrieval_method", "unknown"))


def event(item):
    candidate_ids = item.get("candidate_ids", item.get("candidate_submaps"))
    sources = item.get("candidate_sources", {})
    label_available = bool(item.get("coverage_label_available", False))
    return {
        "method": method_for(item),
        "active_submap": item.get("active_submap"),
        "candidate_ids": candidate_ids if isinstance(candidate_ids, list) else None,
        "candidate_sources": sources if isinstance(sources, dict) else {},
        "candidate_covered": item.get("candidate_covered") if label_available else None,
        "top1_correct": item.get("top1_correct") if label_available else None,
        "candidate_coverage_reason": item.get("candidate_coverage_reason", "label_unavailable"),
        "legal_submaps": item.get("legal_submaps", []),
        "none_of_the_above": item.get("none_of_the_above"),
        "coverage_label_available": label_available,
        "evaluation_eligible": bool(item.get("evaluation_eligible", item.get("state") == "tracking")),
        "geometry_verified": item.get("geometry_verified"),
        "geometry_verification_available": "geometry_verified" in item,
        "reject_reason": item.get("reject_reason", "unavailable"),
        "visual_reliability": item.get("visual_reliability"),
        "pose_reliability": item.get("pose_reliability"),
        "topology_reliability": item.get("topology_reliability"),
        "q_cov": item.get("q_cov"),
        "q_top1": item.get("q_top1"),
        "calibration_version": item.get("calibration_version", "unfitted-v0"),
        "calibration_stage": item.get("calibration_stage", "unfitted"),
        "source_file": item["_source_file"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events = [event(item) for item in iter_status(args.status_root)]
    with (output / "retrieval_events.jsonl").open("w", encoding="utf-8") as handle:
        for item in events:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    with (output / "retrieval_events.csv").open("w", newline="", encoding="utf-8") as handle:
        if events:
            writer = csv.DictWriter(handle, fieldnames=events[0].keys())
            writer.writeheader()
            for item in events:
                writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                                 for key, value in item.items()})

    eligible = [item for item in events if item["evaluation_eligible"]]
    calibrated = [item for item in eligible if item.get("calibration_stage") == "confidence"]
    top1_labeled = [item for item in calibrated
                    if item["top1_correct"] is not None and item.get("q_top1") is not None]
    coverage_labeled = [item for item in calibrated
                        if item["candidate_covered"] is not None and item.get("q_cov") is not None]
    metrics = {
        "status": "available" if top1_labeled and coverage_labeled else "unavailable",
        "reason": ("requires independent top1/coverage labels and calibrated q_top1/q_cov"
                   if not top1_labeled or not coverage_labeled else ""),
        "top1": {"sample_count": len(top1_labeled), "ece": None, "brier": None,
                 "nll": None, "risk_coverage": None},
        "coverage": {"sample_count": len(coverage_labeled), "ece": None,
                     "brier": None, "nll": None},
        "coverage_label_available_fraction": (sum(item["coverage_label_available"] for item in eligible) / len(eligible)
                                               if eligible else 0.0),
        "geometry_verification_available_fraction": (sum(item["geometry_verification_available"] for item in events) / len(events)
                                                     if events else 0.0),
        "calibrated_event_fraction": len(calibrated) / len(events) if events else 0.0,
    }
    if top1_labeled:
        labels = [item["top1_correct"] for item in top1_labeled]
        confidences = [item["q_top1"] for item in top1_labeled]
        metrics["top1"].update(ece=ece(labels, confidences), brier=brier_score(labels, confidences),
                               nll=nll(labels, confidences), risk_coverage=risk_coverage_curve(labels, confidences))
        with (output / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["coverage", "risk", "count"])
            writer.writeheader()
            writer.writerows(metrics["top1"]["risk_coverage"]["points"])
    else:
        (output / "risk_coverage.csv").write_text("coverage,risk,count\n", encoding="utf-8")
    if coverage_labeled:
        labels = [item["candidate_covered"] for item in coverage_labeled]
        confidences = [item["q_cov"] for item in coverage_labeled]
        metrics["coverage"].update(ece=ece(labels, confidences), brier=brier_score(labels, confidences),
                                   nll=nll(labels, confidences))
    (output / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "status_root": str(Path(args.status_root).resolve()),
        "events": len(events),
        "methods": {method: sum(item["method"] == method for item in events) for method in ("c1", "c4", "p", "unknown")},
        "reject_reasons": {},
        "candidate_coverage": None if not any(item["coverage_label_available"] for item in events) else "see retrieval_events.jsonl",
        "calibration": metrics,
    }
    for item in events:
        reason = item["reject_reason"]
        summary["reject_reasons"][reason] = summary["reject_reasons"].get(reason, 0) + 1
    (output / "retrieval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
