#!/usr/bin/env python3
"""Automated observability comparison suite for the turn-recovery study.

Runs the same headless experiment once per observability mode, then attributes
every run with tools/turn_attribution.py and writes one comparison table.

It is resumable: a mode is skipped when its run directory already contains
summary.json (use --force to redo). Everything runs headless (--no-gui).

Run inside WSL with the ROS environment sourced:

  cd /mnt/d/科研slam
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 tools/turn_suite.py --modes b0_plain b1_flag b2_dual \
      --duration 340 --root experiments/vo_observability_suite_20261016

Outputs, per mode, under --root:
  <mode>/                       experiment run directory produced by the runner
  logs/<mode>.log               runner stdout/stderr
  attribution/<mode>_attribution.json
  attribution/<mode>_timeline.png
and at the root:
  suite_summary.json, suite_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MODES = ("b0_plain", "b1_flag", "b2_dual")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(command: List[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return process.returncode


def experiment_command(args, mode: str, output: Path) -> List[str]:
    command = [
        "ros2", "run", "visual_slam_gazebo", "run_experiment_profile.py",
        "--profile", args.profile,
        "--modes", "c0",
        "--runs", "1",
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
    command = [
        sys.executable, str(repository_root() / "tools" / "turn_attribution.py"),
        str(run_dir), "--output", str(output_dir), "--labels", label,
    ]
    if no_plot:
        command.append("--no-plot")
    return command


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def summarize(records: List[Dict[str, Any]]) -> str:
    lines = ["# 可观测性对照套件汇总", ""]
    lines.append("| 指标 | " + " | ".join(record["mode"] for record in records) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in records) + " |")

    def row(label: str, values: List[str]) -> None:
        lines.append("| " + label + " | " + " | ".join(values) + " |")

    def number(record: Dict[str, Any], key: str, digits: int = 2, default: str = "-") -> str:
        report = record.get("report") or {}
        value = report.get(key)
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return ("%." + str(digits) + "f") % value
        return str(value)

    row("运行目录", [record.get("run_dir", "-") for record in records])
    row("状态", ["完成" if record.get("ok") else "失败" for record in records])
    row("帧数", [number(record, "frames", 0) for record in records])
    row("时长 s", [number(record, "span_s", 1) for record in records])
    row("真值 yaw 路径长度", [number(record, "truth_yaw_path_deg") for record in records])
    row("真值 yaw 最大偏离", [number(record, "truth_yaw_max_deg") for record in records])
    row("视觉 yaw 路径长度", [number(record, "visual_yaw_path_deg") for record in records])
    row("视觉 yaw 最大偏离", [number(record, "visual_yaw_max_deg") for record in records])
    row("最长连续空窗 s", [number(record, "blind_window_s", 1) for record in records])
    row("空窗真值转角", [number(record, "blind_truth_yaw_deg") for record in records])
    row("空窗视觉记账", [number(record, "blind_applied_yaw_deg") for record in records])
    row("参考重置丢弃转角", [number(record, "dropped_truth_yaw_deg") for record in records])
    row("应用增量区间数", [number(record, "applied_intervals", 0) for record in records])

    def interval_text(record: Dict[str, Any]) -> str:
        report = record.get("report") or {}
        intervals = report.get("unobserved_intervals") or []
        if not intervals:
            return "0"
        parts = []
        for interval in intervals[:3]:
            closure = interval.get("closure_rotation_deg")
            closure_text = "未闭合" if closure is None else ("%.1f°" % closure)
            parts.append("%d 帧/真值 %.1f°/%s" % (
                interval.get("frames", 0), interval.get("truth_yaw_deg", 0.0), closure_text))
        return "; ".join(parts)

    row("显式 UNOBSERVED 区间", [interval_text(record) for record in records])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--root", required=True, help="suite output root (relative to the repository or absolute)")
    parser.add_argument("--duration", type=float, default=340.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--max-turn-angular", type=float, default=0.1)
    parser.add_argument("--profile", default="main_45m")
    parser.add_argument("--waypoint-route", default="experiments/vo_multi_turn_yaw_slow_010_first_turn_20261016/smoke_routes/mapping_prefix_17.0m.csv")
    parser.add_argument("--no-gui", dest="no_gui", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="rerun modes that already have a summary")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = repository_root() / root
    root.mkdir(parents=True, exist_ok=True)
    attribution_dir = root / "attribution"
    attribution_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for mode in args.modes:
        output = root / mode
        run_dir = output / "c0" / "run_01"
        summary_path = run_dir / "summary.json"
        record: Dict[str, Any] = {"mode": mode, "run_dir": str(run_dir), "ok": False}

        if summary_path.exists() and not args.force:
            print("[suite] " + mode + ": reuse existing run (summary.json present)", flush=True)
            record["ok"] = True
            record["reused"] = True
        else:
            if run_dir.exists() and args.force:
                print("[suite] " + mode + ": --force set, rerunning", flush=True)
            command = experiment_command(args, mode, output)
            log_path = root / "logs" / (mode + ".log")
            started = time.time()
            print("[suite] " + mode + ": running experiment ...", flush=True)
            code = run_command(command, log_path, repository_root())
            elapsed = time.time() - started
            print("[suite] " + mode + ": experiment exit=%d, %.1f s, log=%s" % (code, elapsed, log_path), flush=True)
            record["experiment_exit"] = code
            record["experiment_seconds"] = round(elapsed, 1)
            record["log"] = str(log_path)
            record["ok"] = code == 0 and summary_path.exists()
            if not record["ok"]:
                records.append(record)
                continue

        if run_dir.exists():
            code = run_command(attribution_command(run_dir, attribution_dir, mode, args.no_plot),
                               attribution_dir / (mode + "_attribution.log"), repository_root())
            report_path = attribution_dir / (mode + "_attribution.json")
            record["report"] = load_json(report_path)
            record["attribution_exit"] = code
        records.append(record)

    summary_json = root / "suite_summary.json"
    summary_json.write_text(json.dumps({"modes": args.modes, "records": records}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    markdown = summarize(records)
    (root / "suite_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("[suite] done, outputs in " + str(root))
    return 0 if all(record.get("ok") for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
