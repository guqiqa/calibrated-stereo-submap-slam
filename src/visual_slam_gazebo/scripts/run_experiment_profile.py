#!/usr/bin/env python3
"""Run a named scale experiment from setup through result archival."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GAZEBO_LOCK_PATH = Path("/tmp/visual_slam_gazebo_experiment.lock")


# Edit this table when adding a new route or changing an experiment campaign.
PROFILES = {
    "script_smoke": {
        "route_m": 10.0,
        "duration_s": 50.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/script_smoke",
    },
    "baseline_30m_speed02": {
        "route_m": 30.0,
        "duration_s": 150.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/valid_route_gsp_10x_30m_speed02",
    },
    "main_45m": {
        "route_m": 45.0,
        "duration_s": 225.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/main_45m_gsp",
    },
    "basic_map_45m_roundtrip": {
        "route_m": 45.0,
        "duration_s": 450.0,
        "reverse_after_s": 225.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/basic_map_45m_roundtrip",
    },
    "highload_60m": {
        "route_m": 60.0,
        "duration_s": 300.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/highload_60m_gsp",
    },
    "highload_90m": {
        "route_m": 90.0,
        "duration_s": 450.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/highload_90m_gsp",
    },
    "highload_120m": {
        "route_m": 120.0,
        "duration_s": 600.0,
        "command_speed_mps": 0.254522061,
        "target_actual_speed_mps": 0.2,
        "output": "experiments/highload_120m_gsp",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="main_45m")
    parser.add_argument("--list-profiles", action="store_true",
                        help="List the selectable experiment profiles and exit")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--modes", nargs="+",
                        choices=["c0", "c1", "c4", "p", "global", "fixed", "joint"],
                        default=["global", "fixed", "joint"])
    parser.add_argument("--speed", type=float, default=None,
                        help="Override calibrated command speed")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override profile duration, mainly for smoke tests")
    parser.add_argument("--reverse-after", type=float, default=None,
                        help="Override route reversal time for round-trip profiles")
    parser.add_argument("--angular", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--output-root", default=None,
                        help="Override the profile output directory")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--no-launch-gazebo", action="store_true",
                        help="Require an already running Gazebo instance")
    parser.add_argument("--no-gui", action="store_true",
                        help="Launch Gazebo without the GUI")
    parser.add_argument("--gui", action="store_true",
                        help="Explicitly request the Gazebo GUI (default)")
    parser.add_argument("--record", action="store_true",
                        help="Record a rosbag for every mode/run")
    parser.add_argument("--shutdown-gazebo", action="store_true",
                        help="Close Gazebo when this campaign finishes")
    parser.add_argument("--world", default=None,
                        help="Use a generated Gazebo world instead of the default corridor")
    parser.add_argument("--scene-manifest", default=None,
                        help="Scene manifest; records scene identity and resolves scene.world")
    parser.add_argument("--waypoint-route", default=None,
                        help="Generated mapping/query route CSV")
    return parser.parse_args()


def stop_process(process):
    """Stop a process group without sending terminal key events to Gazebo."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def service_available(service):
    result = subprocess.run(
        ["ros2", "service", "list"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return service in result.stdout.split()


def gazebo_server_pids():
    """Return live Gazebo Classic server PIDs without relying on ROS discovery."""
    result = subprocess.run(
        ["pgrep", "-x", "gzserver"], check=False, capture_output=True, text=True
    )
    return [int(value) for value in result.stdout.split() if value.isdigit()]


def acquire_experiment_lock():
    """Prevent concurrent campaign scripts from each launching Gazebo."""
    handle = GAZEBO_LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another experiment owns {GAZEBO_LOCK_PATH}; stop it before starting a new campaign"
        ) from exc
    handle.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    handle.flush()
    return handle


def require_single_gazebo(expected_pid=None):
    pids = gazebo_server_pids()
    if len(pids) != 1:
        raise RuntimeError(
            f"Expected exactly one gzserver, found {len(pids)}: {pids}. "
            "Stop residual Gazebo processes before running the experiment."
        )
    if expected_pid is not None and pids[0] != expected_pid:
        raise RuntimeError(
            f"Gazebo ownership mismatch: launched PID {expected_pid}, live PID is {pids[0]}"
        )
    return pids[0]


def start_gazebo(output, gui, world=None):
    """Start Gazebo in a detached process group so the terminal stays usable."""
    log_path = output / "gazebo.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        "ros2", "launch", "visual_slam_gazebo",
        "stereo_terminal.launch.py", f"gui:={'true' if gui else 'false'}",
    ]
    if world:
        command.append(f"world:={world}")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, log_handle, log_path


def gui_available():
    result = subprocess.run(
        ["pgrep", "-x", "gzclient"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def start_gazebo_gui(output):
    """Start only the GUI when a Gazebo server is already running."""
    log_path = output / "gzclient.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["gzclient"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, log_handle, log_path


def wait_for_gazebo(process, timeout=45.0):
    deadline = time.monotonic() + timeout
    required = ("/reset_world", "/unpause_physics")
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "Gazebo launch exited before ROS services became available; "
                "see gazebo.log"
            )
        pids = gazebo_server_pids()
        if len(pids) > 1:
            raise RuntimeError(f"Multiple gzserver instances detected during startup: {pids}")
        if all(service_available(name) for name in required):
            require_single_gazebo()
            return
        time.sleep(1.0)
    raise RuntimeError(
        "Timed out waiting for Gazebo services: " + ", ".join(required)
    )


def write_campaign_state(path, state, profile, args, output, gazebo_started,
                         log_path=None, gui_log_path=None):
    payload = {
        "state": state,
        "profile": profile,
        "route_m": PROFILES[profile]["route_m"],
        "modes": args.modes,
        "runs": args.runs,
        "output": str(output),
        "gazebo_started_by_script": gazebo_started,
        "gazebo_gui": not args.no_gui,
        "gazebo_log": str(log_path) if log_path else None,
        "gzclient_log": str(gui_log_path) if gui_log_path else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main():
    args = parse_args()
    scene_manifest = None
    if args.scene_manifest:
        manifest_path = Path(args.scene_manifest).resolve()
        scene_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.world is None:
            args.world = str(manifest_path.parent / "scene.world")
        if not Path(args.world).exists():
            raise SystemExit(f"generated scene world does not exist: {args.world}")
    if args.list_profiles:
        print(json.dumps(PROFILES, indent=2, ensure_ascii=True))
        return
    if args.runs < 1 or args.warmup < 0.0:
        raise SystemExit("runs must be >= 1 and warmup must be >= 0")
    if args.duration is not None and args.duration <= 0.0:
        raise SystemExit("duration must be > 0")
    experiment_lock = acquire_experiment_lock()
    profile = dict(PROFILES[args.profile])
    output = Path(args.output_root or profile["output"]).resolve()
    command_speed = (profile.get("command_speed_mps", 0.2)
                     if args.speed is None else args.speed)
    duration = profile["duration_s"] if args.duration is None else args.duration
    if args.reverse_after is not None:
        reverse_after = args.reverse_after
    elif "reverse_after_s" in profile and args.duration is not None:
        reverse_fraction = profile["reverse_after_s"] / profile["duration_s"]
        reverse_after = duration * reverse_fraction
    else:
        reverse_after = profile.get("reverse_after_s")
    output.mkdir(parents=True, exist_ok=True)
    package_dir = Path(__file__).resolve().parent
    recorder = package_dir / "run_submap_experiment.py"
    command = [
        sys.executable, str(recorder),
        "--modes", *args.modes,
        "--runs", str(args.runs),
        "--duration", str(duration),
        "--warmup", str(args.warmup),
        "--speed", str(command_speed),
        "--angular", str(args.angular),
        "--output", str(output),
        "--export-map",
    ]
    if reverse_after is not None:
        command.extend(["--reverse-after", str(reverse_after)])
    if args.record:
        command.append("--record-bag")
    if args.waypoint_route:
        command.extend(["--waypoint-route", str(Path(args.waypoint_route).resolve())])
    metadata = {
        "profile": args.profile,
        "route_m": profile["route_m"],
        "duration_s": duration,
        "reverse_after_s": reverse_after,
        "modes": args.modes,
        "runs": args.runs,
        "command_speed_mps": command_speed,
        "target_actual_speed_mps": profile.get("target_actual_speed_mps"),
        "speed_calibration": "experiments/speed_calibration_20260905.json",
        "angular_radps": args.angular,
        "scene_model": "154 m maximum scene: 0-120 m evaluation route + approximately 30 m visual buffer",
    }
    if scene_manifest:
        metadata.update({
            "scene_model": scene_manifest.get("scene_id"),
            "scene_manifest": str(Path(args.scene_manifest).resolve()),
            "scene_map_version": scene_manifest.get("map_version"),
            "scene_world": str(Path(args.world).resolve()),
        })
    if args.waypoint_route:
        metadata["waypoint_route"] = str(Path(args.waypoint_route).resolve())
    (output / "profile.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=True), flush=True)

    campaign_state = output / "campaign_state.json"
    gazebo_process = None
    gazebo_log_handle = None
    gazebo_started = False
    gazebo_log_path = None
    gui_process = None
    gui_log_handle = None
    gui_log_path = None
    write_campaign_state(
        campaign_state, "starting", args.profile, args, output, False
    )
    try:
        existing_pids = gazebo_server_pids()
        if len(existing_pids) > 1:
            raise RuntimeError(f"Multiple gzserver instances already running: {existing_pids}")
        if existing_pids:
            if args.world:
                raise RuntimeError(
                    "A gzserver is already running while a generated world was requested. "
                    "Stop it first so the manifest cannot be paired with the wrong world."
                )
            # A server process can exist before DDS services become visible. Wait for
            # that process instead of racing it by launching another server.
            wait_for_gazebo(None)
            require_single_gazebo(existing_pids[0])
            print("Using the existing Gazebo instance.", flush=True)
            if not args.no_gui and not gui_available():
                gui_process, gui_log_handle, gui_log_path = start_gazebo_gui(output)
                print(f"Started the Gazebo GUI; log: {gui_log_path}", flush=True)
        elif args.no_launch_gazebo:
            raise RuntimeError(
                "Gazebo services are unavailable and --no-launch-gazebo was set"
            )
        else:
            gazebo_process, gazebo_log_handle, gazebo_log_path = start_gazebo(
                output, not args.no_gui, args.world
            )
            gazebo_started = True
            write_campaign_state(
                campaign_state, "waiting_for_gazebo", args.profile, args,
                output, True, gazebo_log_path
            )
            print(
                f"Started Gazebo ({'GUI' if not args.no_gui else 'headless'}); "
                f"log: {gazebo_log_path}", flush=True
            )
            wait_for_gazebo(gazebo_process)
            require_single_gazebo()
        write_campaign_state(
            campaign_state, "running", args.profile, args, output,
            gazebo_started, gazebo_log_path, gui_log_path
        )
        subprocess.run(command, check=True)

        if not args.skip_analysis:
            analyzer = package_dir / "analyze_route_segments.py"
            for mode in args.modes:
                mode_dir = output / mode
                for run_dir in sorted(mode_dir.glob("run_*")):
                    if not (run_dir / "summary.json").exists():
                        continue
                    subprocess.run([
                        sys.executable, str(analyzer), str(run_dir),
                        "--segment-size", "5.0",
                        "--max-position", str(profile["route_m"]),
                    ], check=True)
        write_campaign_state(
            campaign_state, "complete", args.profile, args, output,
            gazebo_started, gazebo_log_path, gui_log_path
        )
    except KeyboardInterrupt:
        write_campaign_state(
            campaign_state, "interrupted", args.profile, args, output,
            gazebo_started, gazebo_log_path, gui_log_path
        )
        raise
    except Exception:
        write_campaign_state(
            campaign_state, "failed", args.profile, args, output,
            gazebo_started, gazebo_log_path, gui_log_path
        )
        raise
    finally:
        if args.shutdown_gazebo:
            stop_process(gazebo_process)
            stop_process(gui_process)
        if gazebo_log_handle is not None:
            gazebo_log_handle.close()
        if gui_log_handle is not None:
            gui_log_handle.close()
        experiment_lock.close()


if __name__ == "__main__":
    main()
