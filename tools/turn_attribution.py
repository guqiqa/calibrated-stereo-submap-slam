#!/usr/bin/env python3
"""Offline turn attribution for stereo VO experiment runs.

Reads a run directory (or several) and reports how the accumulated visual yaw
compares with simulation truth, where the error is created, and - when the run
was recorded with the observability state fields - how the explicit
UNOBSERVED intervals line up with the real rotation.

Read-only: no experiment file is ever modified. All outputs go to --output.

Usage:
  python3 tools/turn_attribution.py RUN_DIR [RUN_DIR ...] --output OUT_DIR
  python3 tools/turn_attribution.py RUN_DIR --label b0 --output OUT_DIR

Typical run directory:
  experiments/<campaign>/<mode>/run_01
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= TWO_PI
    while angle < -math.pi:
        angle += TWO_PI
    return angle


def degrees(radians: float) -> float:
    return radians * 180.0 / math.pi


def unwrap_series(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        result.append(result[-1] + wrap_angle(value - result[-1]))
    return result


def quaternion_to_yaw_gazebo(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw about the Gazebo Z axis."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def quaternion_to_yaw_optical_y(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw about the optical Y axis, matching the VO yaw convention."""
    return 2.0 * math.atan2(qy, qw)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


@dataclass
class TruthSample:
    t: float
    yaw_gazebo: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class FrameRow:
    t: float
    raw: Dict[str, str]

    def get(self, key: str, default: Any = None) -> Any:
        value = self.raw.get(key, default)
        return value


@dataclass
class UnobservedInterval:
    interval_id: int
    start_t: float
    end_t: Optional[float]
    frames: int
    truth_yaw_deg: float
    r_pre_frame: int = -1
    r_post_frame: int = -1
    closure_rotation_deg: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "start_t": round(self.start_t, 3),
            "end_t": None if self.end_t is None else round(self.end_t, 3),
            "frames": self.frames,
            "truth_yaw_deg": round(self.truth_yaw_deg, 2),
            "r_pre_frame": self.r_pre_frame,
            "r_post_frame": self.r_post_frame,
            "closure_rotation_deg": None if not math.isfinite(self.closure_rotation_deg)
            else round(self.closure_rotation_deg, 2),
        }


@dataclass
class RunReport:
    run_dir: str
    label: str
    frames: int = 0
    span_s: float = 0.0
    truth_yaw_deg: float = 0.0
    visual_yaw_deg: float = 0.0
    deficit_deg: float = 0.0
    yaw_source: str = "rotation_total_yaw_deg"
    blind_frames: int = 0
    blind_window_s: float = 0.0
    blind_truth_yaw_deg: float = 0.0
    blind_applied_yaw_deg: float = 0.0
    applied_intervals: int = 0
    applied_yaw_deg: float = 0.0
    applied_truth_yaw_deg: float = 0.0
    dropped_events: int = 0
    dropped_truth_yaw_deg: float = 0.0
    by_decision: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_visual_state: Dict[str, Dict[str, float]] = field(default_factory=dict)
    intervals: List[UnobservedInterval] = field(default_factory=list)
    notices: List[str] = field(default_factory=list)
    has_observability: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "label": self.label,
            "frames": self.frames,
            "span_s": round(self.span_s, 2),
            "truth_yaw_deg": round(self.truth_yaw_deg, 2),
            "visual_yaw_deg": round(self.visual_yaw_deg, 2),
            "deficit_deg": round(self.deficit_deg, 2),
            "yaw_source": self.yaw_source,
            "blind_frames": self.blind_frames,
            "blind_window_s": round(self.blind_window_s, 2),
            "blind_truth_yaw_deg": round(self.blind_truth_yaw_deg, 2),
            "blind_applied_yaw_deg": round(self.blind_applied_yaw_deg, 2),
            "applied_intervals": self.applied_intervals,
            "applied_yaw_deg": round(self.applied_yaw_deg, 2),
            "applied_truth_yaw_deg": round(self.applied_truth_yaw_deg, 2),
            "dropped_events": self.dropped_events,
            "dropped_truth_yaw_deg": round(self.dropped_truth_yaw_deg, 2),
            "by_decision": self.by_decision,
            "by_visual_state": self.by_visual_state,
            "unobserved_intervals": [item.as_dict() for item in self.intervals],
            "has_observability": self.has_observability,
            "notices": self.notices,
        }


def load_truth(run_dir: Path) -> List[TruthSample]:
    for name in ("trajectory_truth.csv", "sim_odom.csv"):
        path = run_dir / name
        if not path.exists():
            continue
        samples: List[TruthSample] = []
        for row in read_csv_rows(path):
            t = to_float(row.get("host_time"))
            if not math.isfinite(t):
                continue
            samples.append(TruthSample(
                t=t,
                yaw_gazebo=quaternion_to_yaw_gazebo(
                    to_float(row.get("qx"), 0.0), to_float(row.get("qy"), 0.0),
                    to_float(row.get("qz"), 0.0), to_float(row.get("qw"), 1.0)),
                x=to_float(row.get("x"), 0.0),
                y=to_float(row.get("y"), 0.0),
            ))
        samples.sort(key=lambda item: item.t)
        return samples
    return []


def load_visual_yaw(run_dir: Path) -> Tuple[str, List[Tuple[float, float]]]:
    path = run_dir / "visual_odom.csv"
    if not path.exists():
        return "rotation_total_yaw_deg", []
    series: List[Tuple[float, float]] = []
    for row in read_csv_rows(path):
        t = to_float(row.get("host_time"))
        if not math.isfinite(t):
            continue
        series.append((t, quaternion_to_yaw_optical_y(
            to_float(row.get("qx"), 0.0), to_float(row.get("qy"), 0.0),
            to_float(row.get("qz"), 0.0), to_float(row.get("qw"), 1.0))))
    return "visual_odom.csv", series


def build_report(run_dir: Path, label: str, blind_match_threshold: float) -> RunReport:
    report = RunReport(run_dir=str(run_dir), label=label)
    diagnostics_path = run_dir / "frame_diagnostics.csv"
    if not diagnostics_path.exists():
        report.notices.append("frame_diagnostics.csv not found")
        return report

    truth = load_truth(run_dir)
    if not truth:
        report.notices.append("no truth trajectory found (trajectory_truth.csv / sim_odom.csv)")
        return report
    truth_times = [item.t for item in truth]
    truth_yaw = unwrap_series([item.yaw_gazebo for item in truth])

    def truth_yaw_at(t: float) -> float:
        index = bisect.bisect_left(truth_times, t)
        if index <= 0:
            return truth_yaw[0]
        if index >= len(truth_times):
            return truth_yaw[-1]
        before, after = index - 1, index
        if abs(truth_times[before] - t) <= abs(truth_times[after] - t):
            return truth_yaw[before]
        return truth_yaw[after]

    frames = [FrameRow(t=to_float(row.get("host_time")), raw=row) for row in read_csv_rows(diagnostics_path)]
    frames = [frame for frame in frames if math.isfinite(frame.t)]
    if len(frames) < 2:
        report.notices.append("not enough diagnostic frames")
        return report
    report.frames = len(frames)
    report.span_s = frames[-1].t - frames[0].t
    report.has_observability = "visual_state" in frames[0].raw

    yaw_source, visual_series = load_visual_yaw(run_dir)
    report.yaw_source = yaw_source
    if visual_series:
        visual_yaw = unwrap_series([item[1] for item in visual_series])
        report.visual_yaw_deg = degrees(visual_yaw[-1] - visual_yaw[0])
        report.applied_yaw_deg = to_float(frames[-1].get("rotation_total_yaw_deg"), 0.0)
    else:
        applied = [to_float(frame.get("rotation_total_yaw_deg"), 0.0) for frame in frames]
        report.visual_yaw_deg = applied[-1] - applied[0]
        report.applied_yaw_deg = report.visual_yaw_deg
    report.truth_yaw_deg = degrees(truth_yaw[-1] - truth_yaw[0])
    report.deficit_deg = report.visual_yaw_deg - report.truth_yaw_deg

    by_decision: Dict[str, Dict[str, float]] = {}
    by_visual_state: Dict[str, Dict[str, float]] = {}
    previous_truth = truth_yaw_at(frames[0].t)
    for frame in frames[1:]:
        current_truth = truth_yaw_at(frame.t)
        delta = current_truth - previous_truth
        previous_truth = current_truth
        state = str(frame.get("state") or "")
        decision = str(frame.get("rotation_decision") or "")
        visual_state = str(frame.get("visual_state") or "")
        key = state + " | " + decision
        entry = by_decision.setdefault(key, {"frames": 0.0, "truth_yaw_deg": 0.0})
        entry["frames"] += 1
        entry["truth_yaw_deg"] += degrees(delta)
        if visual_state:
            entry = by_visual_state.setdefault(visual_state, {"frames": 0.0, "truth_yaw_deg": 0.0})
            entry["frames"] += 1
            entry["truth_yaw_deg"] += degrees(delta)
    report.by_decision = {
        key: {"frames": int(value["frames"]), "truth_yaw_deg": round(value["truth_yaw_deg"], 2)}
        for key, value in sorted(by_decision.items(), key=lambda item: -abs(item[1]["truth_yaw_deg"]))
        if abs(value["truth_yaw_deg"]) > 0.5
    }
    report.by_visual_state = {
        key: {"frames": int(value["frames"]), "truth_yaw_deg": round(value["truth_yaw_deg"], 2)}
        for key, value in sorted(by_visual_state.items())
    }

    applied_sum = 0.0
    applied_truth = 0.0
    applied_count = 0
    dropped_truth = 0.0
    dropped_events = 0
    commit_truth = truth_yaw_at(frames[0].t)
    previous_total = to_float(frames[0].get("rotation_total_yaw_deg"), 0.0)
    previous_updates = to_float(frames[0].get("rotation_updates_total"), 0.0)
    blind_frames: List[FrameRow] = []
    for frame in frames[1:]:
        current_truth = truth_yaw_at(frame.t)
        current_total = to_float(frame.get("rotation_total_yaw_deg"), previous_total)
        current_updates = to_float(frame.get("rotation_updates_total"), previous_updates)
        matches = to_float(frame.get("temporal_matches_2d"), float("nan"))
        if math.isfinite(matches) and matches < blind_match_threshold:
            blind_frames.append(frame)
        if current_updates > previous_updates:
            applied_sum += current_total - previous_total
            applied_truth += degrees(current_truth - commit_truth)
            applied_count += 1
            commit_truth = current_truth
        else:
            state = str(frame.get("state") or "")
            decision = str(frame.get("rotation_decision") or "")
            reset = state in {"tracking_lost", "motion_outlier"} or "rebootstrap" in decision
            if reset:
                dropped_truth += degrees(current_truth - commit_truth)
                dropped_events += 1
                commit_truth = current_truth
        previous_total = current_total
        previous_updates = current_updates
    report.applied_intervals = applied_count
    report.applied_yaw_deg = applied_sum
    report.applied_truth_yaw_deg = applied_truth
    report.dropped_events = dropped_events
    report.dropped_truth_yaw_deg = dropped_truth

    if blind_frames:
        report.blind_frames = len(blind_frames)
        report.blind_window_s = blind_frames[-1].t - blind_frames[0].t
        report.blind_truth_yaw_deg = degrees(truth_yaw_at(blind_frames[-1].t) - truth_yaw_at(blind_frames[0].t))
        first_total = to_float(blind_frames[0].get("rotation_total_yaw_deg"), 0.0)
        last_total = to_float(blind_frames[-1].get("rotation_total_yaw_deg"), first_total)
        report.blind_applied_yaw_deg = last_total - first_total

    if report.has_observability:
        current: Optional[UnobservedInterval] = None
        previous_state = str(frames[0].get("visual_state") or "")
        interval_start_truth = 0.0
        for frame in frames[1:]:
            state = str(frame.get("visual_state") or "")
            if state == "UNOBSERVED" and previous_state != "UNOBSERVED":
                current = UnobservedInterval(
                    interval_id=int(to_float(frame.get("unobserved_interval_id"), 0.0)),
                    start_t=frame.t, end_t=None, frames=0,
                    truth_yaw_deg=0.0,
                    r_pre_frame=int(to_float(frame.get("r_pre_frame"), -1.0)))
                interval_start_truth = truth_yaw_at(frame.t)
            if state == "UNOBSERVED" and current is not None:
                current.frames += 1
                current.truth_yaw_deg = degrees(truth_yaw_at(frame.t) - interval_start_truth)
            if state == "REACQUIRED" and current is not None:
                current.end_t = frame.t
                current.r_post_frame = int(to_float(frame.get("r_post_frame"), -1.0))
                current.closure_rotation_deg = to_float(frame.get("unobserved_closure_rotation_deg"))
                report.intervals.append(current)
                current = None
            previous_state = state
        if current is not None:
            report.intervals.append(current)

    return report


def render_markdown(reports: Sequence[RunReport]) -> str:
    lines: List[str] = ["# 转弯归因报告", ""]
    for report in reports:
        lines.append("## " + report.label)
        lines.append("")
        lines.append("- 运行目录：" + report.run_dir)
        lines.append("- 帧数 %d，时长 %.1f s，视觉 yaw 来源：%s" % (report.frames, report.span_s, report.yaw_source))
        lines.append("- 真值累计 yaw **%.2f°**，视觉累计 yaw **%.2f°**，净差 **%+.2f°**" % (
            report.truth_yaw_deg, report.visual_yaw_deg, report.deficit_deg))
        lines.append("- 应用增量区间 %d 段：累计应用 %.2f°，同段真值 %.2f°" % (
            report.applied_intervals, report.applied_yaw_deg, report.applied_truth_yaw_deg))
        lines.append("- 参考重置丢弃：%d 次，丢弃真值转角 %.2f°" % (
            report.dropped_events, report.dropped_truth_yaw_deg))
        lines.append("- 匹配空窗：%d 帧 / %.1f s，期间真值转角 %.2f°，视觉记账 %.2f°" % (
            report.blind_frames, report.blind_window_s,
            report.blind_truth_yaw_deg, report.blind_applied_yaw_deg))
        if report.has_observability:
            lines.append("- 显式 UNOBSERVED 区间：%d 个" % len(report.intervals))
            for interval in report.intervals:
                closure = "未闭合" if not math.isfinite(interval.closure_rotation_deg) else "%.2f°" % interval.closure_rotation_deg
                lines.append("  - #%d 起于 host_time %.3f，持续 %d 帧，真值转角 %.2f°，R_pre=%d → R_post=%d，闭合 %s" % (
                    interval.interval_id, interval.start_t, interval.frames,
                    interval.truth_yaw_deg, interval.r_pre_frame, interval.r_post_frame, closure))
        else:
            lines.append("- 该运行使用 b0_plain，没有显式 UNOBSERVED 区间记录")
        if report.notices:
            lines.append("- 提示：" + "；".join(report.notices))
        lines.append("")
        if report.by_decision:
            lines.append("| state \\| decision | 帧数 | 真值转角(°) |")
            lines.append("| --- | ---: | ---: |")
            for key, value in report.by_decision.items():
                lines.append("| %s | %d | %.2f |" % (key, value["frames"], value["truth_yaw_deg"]))
            lines.append("")
        if report.by_visual_state:
            lines.append("| visual_state | 帧数 | 真值转角(°) |")
            lines.append("| --- | ---: | ---: |")
            for key, value in report.by_visual_state.items():
                lines.append("| %s | %d | %.2f |" % (key, value["frames"], value["truth_yaw_deg"]))
            lines.append("")
    return "\n".join(lines)


def render_plot(run_dir: Path, report: RunReport, output: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    diagnostics_path = run_dir / "frame_diagnostics.csv"
    truth = load_truth(run_dir)
    if not diagnostics_path.exists() or not truth:
        return False
    frames = [FrameRow(t=to_float(row.get("host_time")), raw=row) for row in read_csv_rows(diagnostics_path)]
    frames = [frame for frame in frames if math.isfinite(frame.t)]
    if not frames:
        return False
    t0 = frames[0].t
    times = [frame.t - t0 for frame in frames]
    truth_times = [item.t - t0 for item in truth]
    truth_yaw = unwrap_series([degrees(item.yaw_gazebo) for item in truth])

    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(truth_times, truth_yaw, label="truth yaw (deg)", color="black")
    applied = [to_float(frame.get("rotation_total_yaw_deg"), 0.0) for frame in frames]
    axes[0].plot(times, applied, label="applied rotation total (deg)", color="tab:blue")
    axes[0].set_ylabel("yaw (deg)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    matches = [to_float(frame.get("temporal_matches_2d"), 0.0) for frame in frames]
    axes[1].plot(times, matches, color="tab:orange", label="temporal_matches_2d")
    axes[1].set_ylabel("matches")
    axes[1].grid(alpha=0.3)
    if report.has_observability:
        state_colors = {"OBSERVED": "tab:green", "UNOBSERVED": "tab:red", "REACQUIRED": "tab:purple"}
        previous = None
        start = 0.0
        for index, frame in enumerate(frames):
            state = str(frame.get("visual_state") or "")
            if previous is None:
                previous, start = state, times[index]
                continue
            if state != previous:
                axes[1].axvspan(start, times[index], color=state_colors.get(previous, "gray"), alpha=0.15)
                previous, start = state, times[index]
        axes[1].axvspan(start, times[-1], color=state_colors.get(previous, "gray"), alpha=0.15)
        axes[1].legend(loc="upper right")

    ratio = [to_float(frame.get("evidence_inlier_ratio"), float("nan")) for frame in frames]
    view = [to_float(frame.get("evidence_view_change_deg"), float("nan")) for frame in frames]
    if any(math.isfinite(value) for value in ratio):
        axes[2].plot(times, ratio, color="tab:cyan", label="evidence_inlier_ratio")
    if any(math.isfinite(value) for value in view):
        axes[2].plot(times, view, color="tab:red", label="evidence_view_change_deg")
    axes[2].set_ylabel("evidence")
    axes[2].set_xlabel("time since first frame (s)")
    if axes[2].lines:
        axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    figure.suptitle("turn attribution: " + report.label)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="experiment run directories (contain frame_diagnostics.csv)")
    parser.add_argument("--output", required=True, help="output directory for reports and plots")
    parser.add_argument("--labels", nargs="*", default=None, help="optional labels, one per run directory")
    parser.add_argument("--blind-match-threshold", type=float, default=10.0,
                        help="temporal_matches_2d below this value counts as a blind frame")
    parser.add_argument("--no-plot", action="store_true", help="skip PNG rendering")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or []
    reports: List[RunReport] = []
    for index, run_dir_arg in enumerate(args.run_dirs):
        run_dir = Path(run_dir_arg)
        label = labels[index] if index < len(labels) else (run_dir.parent.name + "/" + run_dir.name)
        report = build_report(run_dir, label, args.blind_match_threshold)
        reports.append(report)
        (output_dir / (label.replace("/", "_") + "_attribution.json")).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_plot:
            plot_ok = render_plot(run_dir, report, output_dir / (label.replace("/", "_") + "_timeline.png"))
            if not plot_ok:
                report.notices.append("plot skipped (matplotlib unavailable or data missing)")

    markdown = render_markdown(reports)
    (output_dir / "turn_attribution.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("outputs written to " + str(output_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
