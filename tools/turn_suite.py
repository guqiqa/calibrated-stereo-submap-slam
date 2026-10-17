#!/usr/bin/env python3
"""Automated observability comparison suite for the turn-recovery study.

Runs the same headless experiment once per observability mode and per repeat,
attributes every run with tools/turn_attribution.py, and writes one comparison
table with the spread across repeats.

Resumable: a mode with all requested runs already summarized is skipped unless
--force is given. Everything runs headless (--no-gui).

Run inside WSL with the ROS environment sourced:

  cd /mnt/d/科研slam
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 tools/turn_suite.py --modes b0_plain b2_dual --runs 3 --duration 340 \
      --root experiments/vo_turn_repeat_20261016

Outputs under --root:
  <mode>/c0/run_NN/                 experiment runs produced by the runner
  logs/<mode>.log                   runner stdout/stderr
  attribution/<mode>_runNN_*.json   attribution reports
  suite_summary.json, suite_summary.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MODES = ("b0_plain", "b1_flag", "b2_dual")

NUMERIC_KEYS = [
    ("frames", "frames", 0),
    ("span_s", "duration_s", 1),
    ("truth_yaw_path_deg", "truth_yaw_path_deg", 2),
    ("truth_yaw_max_deg", "truth_yaw_max_deg", 2),
    ("visual_yaw_path_deg", "visual_yaw_path_deg", 2),
    ("visual_yaw_max_deg", "visual_yaw_max_deg", 2),
    ("blind_window_s", "blind_window_s", 1),
    ("blind_truth_yaw_deg", "blind_truth_yaw_deg", 2),
    ("blind_applied_yaw_deg", "blind_applied_yaw_deg", 2),
    ("dropped_truth_yaw_deg", "dropped_truth_yaw_deg", 2),
    ("applied_intervals", "applied_intervals", 0),
    ("applied_yaw_deg", "applied_yaw_deg", 2),
]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        return subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT).returncode


def stop_leftover_gazebo() -> None:
    """Kill stale Gazebo processes so every attempt starts from a fresh world."""
    for pattern in ("gzserver", "gzclient"):
        subprocess.run(["pkill", "-f", pattern], check=False)
    time.sleep(1.5)


def experiment_command(args, mode: str, output: Path) -> List[str]:
    command = [
        "ros2", "run", "visual_slam_gazebo", "run_experiment_profile.py",
        "--profile", args.profile,
        "--modes", "c0",
        "--runs", str(args.runs),
        "--duration", str(args.duration),
        "--warmup", str(args.warmup),
        "--max-turn-angular", str(args.max_turn_angular),
        "--observability-mode", mode,
        "--output-root", str(output),
    ]
    if args.no_gui:
        command.append("--no-gui")
    if args.waypoint_route:
        command.extend(["--waypoint-route", str(args.waypoint_route)])
    return command


def attribution_command(run_dir: Path, output_dir: Path, label: str, no_plot: bool) -> List[str]:
    command = [sys.executable, str(repository_root() / "tools" / "turn_attribution.py"),
               str(run_dir), "--output", str(output_dir), "--labels", label]
    if no_plot:
        command.append("--no-plot")
    return command


def valid_run(run_dir: Path, min_status_messages: int) -> bool:
    """A run is analysable only when the VO actually processed frames."""
    summary = load_json(run_dir / "summary.json")
    if not summary:
        return False
    if int(summary.get("status_messages", 0)) < min_status_messages:
        return False
    return summary.get("last_state") != "waiting_for_camera_info"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def format_range(values: List[float], digits: int) -> str:
    if not values:
        return "-"
    pattern = "%." + str(digits) + "f"
    if len(values) == 1:
        return pattern % values[0]
    mean = sum(values) / len(values)
    return (pattern + " [" + pattern + "~" + pattern + "]") % (mean, min(values), max(values))


def interval_text(reports: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for report in reports:
        intervals = report.get("unobserved_intervals") or []
        if not intervals:
            parts.append("0")
            continue
        pieces = []
        for interval in intervals[:2]:
            closure = interval.get("closure_rotation_deg")
            closure_text = "未闭合" if closure is None else ("%.1f" % closure)
            pieces.append("%d帧/%.0f°/%s" % (interval.get("frames", 0),
                                             interval.get("truth_yaw_deg", 0.0), closure_text))
        parts.append("; ".join(pieces))
    return " | ".join(parts)


def summarize(mode_reports: Dict[str, List[Dict[str, Any]]]) -> str:
    modes = list(mode_reports.keys())
    lines = ["# 可观测性对照套件汇总", ""]
    runs = len(next(iter(mode_reports.values()))) if mode_reports else 0
    lines.append("每组重复 %d 次，表中为 均值 [最小~最大]。" % runs)
    lines.append("")
    lines.append("| 指标 | " + " | ".join(modes) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in modes) + " |")
    for key, label, digits in NUMERIC_KEYS:
        cells = []
        for mode in modes:
            values = [report[key] for report in mode_reports[mode]
                      if isinstance(report.get(key), (int, float))]
            cells.append(format_range([float(value) for value in values], digits))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.append("| UNOBSERVED 区间 | " + " | ".join(
        interval_text(mode_reports[mode]) for mode in modes) + " |")
    lines.append("")
    lines.append("b0_plain = 旧行为；b1_flag = 单参考 + 失效标记；b2_dual = 双参考 + 显式 unobserved。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--runs", type=int, default=1, help="repeats per mode")
    parser.add_argument("--root", required=True)
    parser.add_argument("--duration", type=float, default=340.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--max-turn-angular", type=float, default=0.1)
    parser.add_argument("--profile", default="main_45m")
    parser.add_argument("--waypoint-route",
                        default="experiments/vo_multi_turn_yaw_slow_010_first_turn_20261016/smoke_routes/mapping_prefix_17.0m.csv")
    parser.add_argument("--no-gui", dest="no_gui", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="retry the whole mode when too few runs produced data")
    parser.add_argument("--min-status-messages", type=int, default=100,
                        help="a run with fewer VO status messages counts as dead")
    parser.add_argument("--keep-gazebo", action="store_true",
                        help="do not kill stale Gazebo processes before an attempt")
    parser.add_argument("--timeout-min", type=float, default=45.0,
                        help="budget per mode; remaining attempts are skipped past it")
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    root = Path(args.root)
    if not root.is_absolute():
        root = repository_root() / root
    root.mkdir(parents=True, exist_ok=True)
    attribution_dir = root / "attribution"
    attribution_dir.mkdir(parents=True, exist_ok=True)

    mode_reports: Dict[str, List[Dict[str, Any]]] = {}
    for mode in args.modes:
        output = root / mode
        runs_dir = output / "c0"
        attempt = 0
        valid: List[Path] = []
        per_attempt_min = args.runs * (args.duration + 50.0) / 60.0
        deadline = time.time() + args.timeout_min * 60.0
        print("[suite] " + mode + ": budget %.0f min, about %.0f min per attempt, up to %d attempt(s)" % (
            args.timeout_min, per_attempt_min, args.max_attempts), flush=True)
        while attempt < args.max_attempts:
            attempt += 1
            if time.time() > deadline:
                print("[suite] " + mode + ": budget exhausted, stopping", flush=True)
                break
            existing = sorted(runs_dir.glob("run_*")) if runs_dir.exists() else []
            valid = [path for path in existing if valid_run(path, args.min_status_messages)]
            if len(valid) >= args.runs and not args.force:
                print("[suite] " + mode + ": reuse " + str(len(valid)) + " valid run(s)", flush=True)
                break
            if not args.keep_gazebo:
                stop_leftover_gazebo()
            command = experiment_command(args, mode, output)
            log_path = root / "logs" / (mode + ".log")
            started = time.time()
            print("[suite] " + mode + ": attempt %d/%d, running %d repeat(s) ..." % (
                attempt, args.max_attempts, args.runs), flush=True)
            code = run_command(command, log_path, repository_root())
            print("[suite] " + mode + ": exit=%d, %.1f s, log=%s" % (code, time.time() - started, log_path),
                  flush=True)
            existing = sorted((runs_dir.glob("run_*") if runs_dir.exists() else []))
            valid = [path for path in existing if valid_run(path, args.min_status_messages)]
            if len(valid) >= args.runs:
                break
            print("[suite] " + mode + ": only %d valid run(s) of %d, retrying" % (len(valid), args.runs),
                  flush=True)

        reports: List[Dict[str, Any]] = []
        for run_dir in valid[:args.runs]:
            label = mode + "_" + run_dir.name
            run_command(attribution_command(run_dir, attribution_dir, label, args.no_plot),
                        attribution_dir / (label + ".log"), repository_root())
            report = load_json(attribution_dir / (label + "_attribution.json"))
            if report is not None:
                report["run_dir"] = str(run_dir)
                reports.append(report)
        if len(reports) < args.runs:
            print("[suite] " + mode + ": WARNING only " + str(len(reports)) + " analysable run(s)", flush=True)
        mode_reports[mode] = reports

    payload = {"modes": args.modes, "runs": args.runs, "reports": mode_reports}
    (root / "suite_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = summarize(mode_reports)
    (root / "suite_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("[suite] done, outputs in " + str(root))
    return 0 if all(mode_reports.get(mode) for mode in args.modes) else 1


if __name__ == "__main__":
    sys.exit(main())
