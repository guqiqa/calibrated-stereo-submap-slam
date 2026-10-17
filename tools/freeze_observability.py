#!/usr/bin/env python3
"""Pick observability thresholds from recorded runs (T1 helper).

Every frame is labelled by outcome rather than by the state machine: a frame
counts as measured when the pose was actually advanced by an accepted estimate
(a rotation increment or a metric transform). The evidence values are then used
to score candidate thresholds, so the frozen numbers are chosen against real
outcomes instead of being guessed.

Usage:
  python3 tools/freeze_observability.py RUN_DIR [RUN_DIR ...] \
      --output experiments/t1_analysis --labels rate_0.05 rate_0.1 ...

Outputs freeze_report.md and freeze_report.json in --output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

METRIC_SOURCES = {
    "pnp_stereo_consistent", "pnp_quantized_3d3d", "pnp_fallback",
    "stereo_3d3d_fallback",
}

CRITERIA = [
    ("features", "min", "obs_min_features_enter", "obs_min_features_ok"),
    ("inliers", "min", "obs_min_inliers_enter", "obs_min_inliers_ok"),
    ("inlier_ratio", "min", "obs_min_inlier_ratio_enter", "obs_min_inlier_ratio_ok"),
    ("reprojection_px", "max", "obs_max_reprojection_enter", "obs_max_reprojection_ok"),
    ("grid_coverage", "min", "obs_min_grid_coverage_enter", "obs_min_grid_coverage_ok"),
]

CURRENT = {
    "obs_min_features_enter": 15.0, "obs_min_features_ok": 30.0,
    "obs_min_inliers_enter": 10.0, "obs_min_inliers_ok": 20.0,
    "obs_min_inlier_ratio_enter": 0.25, "obs_min_inlier_ratio_ok": 0.45,
    "obs_max_reprojection_enter": 3.0, "obs_max_reprojection_ok": 2.0,
    "obs_min_grid_coverage_enter": 3.0, "obs_min_grid_coverage_ok": 5.0,
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def score_threshold(values: List[float], labels: List[bool], threshold: float,
                    mode: str) -> Tuple[int, int, int, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for value, label in zip(values, labels):
        predicted = (value >= threshold) if mode == "min" else (value <= threshold)
        if label and predicted:
            true_positive += 1
        elif label and not predicted:
            false_negative += 1
        elif not label and predicted:
            false_positive += 1
        else:
            true_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def f1_of(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def best_threshold(values: List[float], labels: List[bool], mode: str,
                   min_recall: float) -> Dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"threshold": float("nan"), "f1": 0.0, "recall": 0.0, "precision": 0.0}
    candidates = sorted(set(finite))
    best: Optional[Dict[str, float]] = None
    for threshold in candidates:
        tp, fp, tn, fn = score_threshold(values, labels, threshold, mode)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        if recall < min_recall:
            continue
        f1 = f1_of(tp, fp, fn)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": f1, "recall": recall, "precision": precision}
    if best is None:
        best = {"threshold": float("nan"), "f1": 0.0, "recall": 0.0, "precision": 0.0}
    return best


def analyse_run(run_dir: Path) -> Dict[str, Any]:
    diagnostics = run_dir / "frame_diagnostics.csv"
    if not diagnostics.exists():
        return {"run_dir": str(run_dir), "notices": ["frame_diagnostics.csv missing"]}
    rows = read_rows(diagnostics)
    labels: List[bool] = []
    columns: Dict[str, List[float]] = {name: [] for name, _, _, _ in CRITERIA}
    previous_updates = None
    for row in rows:
        updates = to_float(row.get("rotation_updates_total"), 0.0)
        source = str(row.get("motion_source") or "")
        measured = False
        if previous_updates is not None and updates > previous_updates:
            measured = True
        if source in METRIC_SOURCES and to_float(row.get("pnp_inliers"), 0.0) > 0:
            measured = True
        previous_updates = updates
        labels.append(measured)
        columns["features"].append(to_float(row.get("evidence_features")))
        columns["inliers"].append(to_float(row.get("evidence_inliers")))
        columns["inlier_ratio"].append(to_float(row.get("evidence_inlier_ratio")))
        columns["reprojection_px"].append(to_float(row.get("evidence_reprojection_px")))
        columns["grid_coverage"].append(to_float(row.get("evidence_grid_coverage")))
    measured_values = {
        name: [value for value, label in zip(columns[name], labels) if label and math.isfinite(value)]
        for name in columns
    }
    unmeasured_values = {
        name: [value for value, label in zip(columns[name], labels) if not label and math.isfinite(value)]
        for name in columns
    }
    return {
        "run_dir": str(run_dir),
        "frames": len(rows),
        "measured": sum(1 for label in labels if label),
        "columns": columns,
        "labels": labels,
        "measured_values": measured_values,
        "unmeasured_values": unmeasured_values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--min-recall", type=float, default=0.90,
                        help="minimum recall when choosing an enter threshold")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    for index, run_dir_arg in enumerate(args.run_dirs):
        label = (args.labels[index] if args.labels and index < len(args.labels)
                 else Path(run_dir_arg).parent.parent.name)
        report = analyse_run(Path(run_dir_arg))
        report["label"] = label
        reports.append(report)

    merged_values: Dict[str, List[float]] = {name: [] for name, _, _, _ in CRITERIA}
    merged_labels: List[bool] = []
    for report in reports:
        for name in merged_values:
            merged_values[name].extend(report["columns"][name])
        merged_labels.extend(report["labels"])

    recommendations: Dict[str, Dict[str, float]] = {}
    rows_markdown: List[str] = [
        "# 可观测性判据冻结分析", "",
        "标签定义：帧的位姿是否被一个合格估计实际推进（旋转增量或度量变换）。", "",
        "| 运行 | 帧数 | 被测量帧 | 占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for report in reports:
        share = report["measured"] / report["frames"] if report["frames"] else 0.0
        rows_markdown.append("| %s | %d | %d | %.1f%% |" % (
            report["label"], report["frames"], report["measured"], 100.0 * share))
    rows_markdown.extend(["", "| 判据 | 被测量帧 p05/p50 | 未测量帧 p50/p95 | 当前 enter | 推荐 enter | 推荐 ok | F1 |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, mode, enter_key, ok_key in CRITERIA:
        values = merged_values[name]
        labels = merged_labels
        measured = report and [value for value, flag in zip(values, labels)
                               if flag and math.isfinite(value)]
        unmeasured = [value for value, flag in zip(values, labels) if not flag and math.isfinite(value)]
        enter = best_threshold(values, labels, mode, args.min_recall)
        ok = best_threshold(values, labels, mode, 0.55)
        recommendations[enter_key] = enter["threshold"]
        recommendations[ok_key] = ok["threshold"]
        rows_markdown.append("| %s | %.3f / %.3f | %.3f / %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            name, percentile(measured, 0.05), percentile(measured, 0.50),
            percentile(unmeasured, 0.50), percentile(unmeasured, 0.95),
            CURRENT.get(enter_key, float("nan")), enter["threshold"], ok["threshold"], enter["f1"]))

    rows_markdown.extend([
        "", "推荐值只作为配置候选：需要人工确认 enter 阈值比 ok 阈值宽松（迟滞），",
        "并且不与既有参数（min_3d3d_inliers / min_pnp_inliers / switch_max_reprojection_px）矛盾。", "",
    ])
    markdown = "\n".join(rows_markdown)
    (output_dir / "freeze_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "freeze_report.json").write_text(json.dumps({
        "labels": [report["label"] for report in reports],
        "frames": {report["label"]: report["frames"] for report in reports},
        "measured": {report["label"]: report["measured"] for report in reports},
        "recommended": recommendations,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
