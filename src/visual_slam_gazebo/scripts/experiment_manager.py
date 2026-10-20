#!/usr/bin/env python3
"""Unified project selection, experiment execution, analysis, and archival."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


METHODS = ("c0", "c1", "c4", "p")
ROUTE_PROFILES = {10: "script_smoke", 45: "main_45m", 60: "highload_60m",
                  90: "highload_90m", 120: "highload_120m"}
PATH_OPTIONS = {"--config", "--output-root", "--scene-manifest", "--waypoint-route",
                "--calibration-model", "--dataset-manifest", "--map-database",
                "--separation-report"}


def windows_to_wsl_path(value):
    raw = str(value)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if not match:
        absolute = str(Path(value).resolve())
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", absolute)
    if not match:
        return value
    drive, remainder = match.groups()
    return f"/mnt/{drive.lower()}/{remainder.replace(chr(92), '/')}"


def dispatch_windows_to_wsl(argv):
    """Run the Linux-only ROS/Gazebo workflow when called from PowerShell."""
    translated = []
    expect_path = False
    for value in argv:
        if expect_path:
            translated.append(windows_to_wsl_path(value))
            expect_path = False
            continue
        translated.append(value)
        expect_path = value in PATH_OPTIONS
    script = windows_to_wsl_path(__file__)
    root = windows_to_wsl_path(repository_root())
    python_command = shlex.join(["python3", script, *translated])
    shell_command = (
        "source /opt/ros/humble/setup.bash && "
        f"if [ -f {shlex.quote(root + '/install/setup.bash')} ]; then "
        f"source {shlex.quote(root + '/install/setup.bash')}; fi && "
        f"exec {python_command}")
    distro = os.environ.get("VISUAL_SLAM_WSL_DISTRO", "Ubuntu-22.04")
    return subprocess.call(["wsl.exe", "-d", distro, "bash", "-lc", shell_command])


def repository_root():
    return Path(__file__).resolve().parents[3]


def default_config():
    source_config = repository_root() / "config" / "experiment_profiles.yaml"
    if source_config.exists():
        return source_config
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("visual_slam_gazebo")) / "config" / "experiment_profiles.yaml"
    except (ImportError, LookupError):
        return source_config


def load_profiles(path):
    # JSON is valid YAML. Keeping the file in this subset avoids a PyYAML runtime dependency.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--list-projects", action="store_true")
    parser.add_argument("--show-project")
    parser.add_argument("--project")
    parser.add_argument("--scenario")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output-root")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--shutdown-gazebo", action="store_true")
    parser.add_argument("--scene-manifest", default=None,
                        help="Generated scene_manifest.json; resolves scene.world")
    parser.add_argument("--waypoint-route", default=None,
                        help="Generated route CSV; overrides the project route")
    parser.add_argument("--route-prefix-m", type=float, default=None,
                        help="Pass a mapping-route prefix to the legacy runner for smoke tests")
    parser.add_argument("--route-waypoint-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Pass an inclusive local waypoint interval to the runner")
    parser.add_argument("--max-turn-angular", type=float, default=None,
                        help="Target delivered angular rate for waypoint turns")
    parser.add_argument("--angular-calibration", default="disabled",
                        choices=("disabled", "in_place_v1", "session_v1"),
                        help="Delivery compensation for angular.z")
    parser.add_argument("--angular-calibration-file", default=None)
    parser.add_argument("--max-delivered-angular", type=float, default=0.3472)
    parser.add_argument("--calibration-model", default=None,
                        help="Frozen calibration_model.json used by method P")
    parser.add_argument("--preflight", action="store_true",
                        help="require the formal Gate A-D preflight to pass after analysis")
    parser.add_argument("--dataset-manifest", default=None,
                        help="frozen full dataset manifest used by formal preflight")
    parser.add_argument("--require-coverage-labels", action="store_true")
    parser.add_argument("--map-database", default=None)
    parser.add_argument("--map-database-mode", choices=("online", "build", "load"), default="online")
    parser.add_argument("--separation-report", default=None)
    return parser.parse_args()


def git_value(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def update_state(path, state, **details):
    payload = {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    write_json(path, payload)


def route_jobs(profile):
    routes = profile.get("routes_m", [profile.get("route_m")])
    jobs = []
    for value in routes:
        route = int(value)
        duration_map = profile.get("duration_s_by_route", {})
        duration = duration_map.get(str(route), profile.get("duration_s"))
        if duration is None:
            raise ValueError(f"project has no duration for route {route}")
        jobs.append((route, float(duration)))
    return jobs


def resolve_scene_input(path, root, scene_root=None):
    """Resolve CLI paths from the repository first, then from the scene directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    repository_candidate = Path(root) / candidate
    if repository_candidate.exists() or scene_root is None:
        return repository_candidate.resolve()
    return (Path(scene_root) / candidate).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_campaign(root, project, output):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    target = root / "paper_package" / f"{project}_{timestamp}"
    for name in ("manifest", "analysis", "logs", "protocol"):
        (target / name).mkdir(parents=True, exist_ok=True)
    for name in ("campaign_manifest.json", "campaign_state.json", "profile.json"):
        source = output / name
        if source.exists():
            shutil.copy2(source, target / "manifest" / name)
    if (output / "analysis").exists():
        shutil.copytree(output / "analysis", target / "analysis", dirs_exist_ok=True)
    for source in output.glob("*.log"):
        shutil.copy2(source, target / "logs" / source.name)
    protocol_dir = root / "src" / "visual_slam_gazebo" / "scripts"
    for name in ("protocol_spec.md", "submap_retrieval_protocol.py", "calibration_metrics.py"):
        if (protocol_dir / name).exists():
            shutil.copy2(protocol_dir / name, target / "protocol" / name)
    index = []
    for source in sorted(output.rglob("*")):
        if source.is_file():
            index.append({"path": str(source.relative_to(output)), "size": source.stat().st_size,
                          "sha256": sha256(source), "complete": True})
    write_json(target / "raw_data_index.json", {"source": str(output), "files": index})
    (target / "README.md").write_text(
        f"# {project} experiment archive\n\nRaw data remains at `{output}`. "
        "This package contains manifests, analysis, logs, protocol snapshots, and a SHA-256 index.\n",
        encoding="utf-8")
    return target


def main():
    args = parse_args()
    profiles = load_profiles(args.config)
    if args.list_projects:
        for name, profile in profiles.items():
            print(f"{name}: {profile.get('scene')} ({'ready' if profile.get('implemented', True) else 'not implemented'})")
        return
    if args.show_project:
        if args.show_project not in profiles:
            raise SystemExit(f"unknown project: {args.show_project}")
        print(json.dumps(profiles[args.show_project], indent=2, ensure_ascii=False))
        return
    if not args.project or args.project not in profiles:
        raise SystemExit("--project must name a configured project; use --list-projects")
    profile = dict(profiles[args.project])
    if not profile.get("implemented", True):
        raise SystemExit(profile.get("blocked_reason", "project is not implemented"))
    if args.scenario and args.scenario != profile.get("scene"):
        raise SystemExit(f"scenario {args.scenario} does not match configured scene {profile.get('scene')}")
    methods = args.methods or profile["methods"]
    calibration_model = Path(args.calibration_model).resolve() if args.calibration_model else None
    map_database = Path(args.map_database).resolve() if args.map_database else None
    if args.map_database_mode in {"build", "load"} and map_database is None:
        raise SystemExit("map database build/load mode requires --map-database")
    if args.map_database_mode == "load" and not map_database.exists():
        raise SystemExit(f"map database does not exist: {map_database}")
    if calibration_model and not calibration_model.exists():
        raise SystemExit(f"calibration model does not exist: {calibration_model}")
    if profile.get("require_calibrated_p") and "p" in methods and calibration_model is None:
        raise SystemExit("this profile requires --calibration-model when method P is selected")
    calibration_version = profile.get("calibration_version", "unfitted-v0")
    if calibration_model:
        calibration_version = json.loads(calibration_model.read_text(encoding="utf-8")).get(
            "version", "unknown")
    runs = args.runs or int(profile["runs"])
    if runs < 1:
        raise SystemExit("runs must be >= 1")
    gui = profile.get("gui", True)
    if args.gui:
        gui = True
    if args.no_gui:
        gui = False
    record = profile.get("record_bag", False)
    if args.record:
        record = True
    if args.no_record:
        record = False

    root = repository_root()
    scene_manifest_path = Path(args.scene_manifest).resolve() if args.scene_manifest else None
    scene_manifest = None
    scene_world = None
    if scene_manifest_path:
        scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
        scene_world = scene_manifest_path.parent / "scene.world"
        if not scene_world.exists():
            raise SystemExit(f"scene manifest has no scene.world: {scene_world}")
        expected = scene_manifest.get("files", {}).get("scene.world", {}).get("sha256")
        if expected and sha256(scene_world) != expected:
            raise SystemExit("scene.world does not match scene_manifest.json SHA-256")
        if (profile.get("scene") == "generated_topology_scene" and
                scene_manifest.get("schema_version") != 2):
            raise SystemExit("generated topology experiments require heterogeneous submap schema_version=2")
    elif profile.get("scene_manifest_required"):
        raise SystemExit(f"project {args.project} requires --scene-manifest")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_root).resolve() if args.output_root else root / "experiments" / f"{args.project}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "campaign_state.json"
    manifest = {
        "project": args.project, "scene": profile["scene"], "scenario_seed": profile.get("scene_seed", 0),
        "methods": methods, "runs": runs, "routes": route_jobs(profile), "speed_mps": profile["speed_mps"],
        "query_mode": profile.get("query_mode"), "record_bag": record, "gui": gui,
        "calibration_version": calibration_version,
        "calibration_model": str(calibration_model) if calibration_model else None,
        "calibration_model_sha256": sha256(calibration_model) if calibration_model else None,
        "map_database": str(map_database) if map_database else None,
        "map_database_mode": args.map_database_mode,
        "map_database_sha256": sha256(map_database) if map_database and map_database.exists() else None,
        "camera": {"width": 640, "height": 480, "rate_hz": 15, "baseline_m": 0.07},
        "submap": {"corridor_length_m": 5.0, "corridor_overlap_m": 1.5,
                   "hall_cell_size_m": 7.0, "hall_overlap_m": 1.5,
                   "junction_radius_m": 3.0, "visual_budget": 4,
                   "pose_budget": 3, "topology_budget": 2, "total_budget": 10},
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(root, "status", "--porcelain")),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if scene_manifest:
        manifest["scene_manifest"] = str(scene_manifest_path)
        manifest["scene_id"] = scene_manifest.get("scene_id")
        manifest["scene_map_version"] = scene_manifest.get("map_version")
        manifest["scene_schema_version"] = scene_manifest.get("schema_version")
        manifest["scene_obstacle_mode"] = scene_manifest.get("obstacle_mode")
        manifest["scene_manifest_sha256"] = sha256(scene_manifest_path)
    write_json(output / "campaign_manifest.json", manifest)
    # JSON is valid YAML and preserves a dependency-free frozen configuration snapshot.
    write_json(output / "resolved_parameters.yaml", {"project": profile, "resolved": manifest})
    runner = root / "src" / "visual_slam_gazebo" / "scripts" / "run_experiment_profile.py"
    analyzer = root / "src" / "visual_slam_gazebo" / "scripts" / "analyze_experiment.py"
    try:
        update_state(state_path, "starting", project=args.project, output=str(output))
        jobs = route_jobs(profile)
        for job_index, (route, configured_duration) in enumerate(jobs):
            duration = args.duration if args.duration is not None else configured_duration
            route_output = output if len(route_jobs(profile)) == 1 else output / f"{route}m"
            profile_name = ROUTE_PROFILES.get(route, "main_45m")
            command = [sys.executable, str(runner), "--profile", profile_name,
                       "--modes", *methods, "--runs", str(runs), "--duration", str(duration),
                       "--output-root", str(route_output),
                       "--skip-analysis"]
            if scene_world:
                command.extend(["--world", str(scene_world), "--scene-manifest", str(scene_manifest_path)])
            waypoint_route = args.waypoint_route or profile.get("waypoint_route")
            if waypoint_route:
                waypoint_path = resolve_scene_input(
                    waypoint_route, root,
                    scene_manifest_path.parent if scene_manifest_path else None)
                if not waypoint_path.exists():
                    raise SystemExit(f"waypoint route does not exist: {waypoint_path}")
                command.extend(["--waypoint-route", str(waypoint_path)])
                if args.route_prefix_m is not None:
                    command.extend(["--route-prefix-m", str(args.route_prefix_m)])
                if args.route_waypoint_range is not None:
                    command.extend(["--route-waypoint-range",
                                    str(args.route_waypoint_range[0]),
                                    str(args.route_waypoint_range[1])])
            if args.max_turn_angular is not None:
                command.extend(["--max-turn-angular", str(args.max_turn_angular)])
            command.extend(["--angular-calibration", args.angular_calibration])
            command.extend(["--max-delivered-angular",
                            str(args.max_delivered_angular)])
            if args.angular_calibration_file:
                command.extend(["--angular-calibration-file",
                                str(args.angular_calibration_file)])
            if calibration_model:
                command.extend(["--calibration-model", str(calibration_model)])
            if args.require_coverage_labels:
                command.append("--require-coverage-labels")
            if map_database:
                command.extend(["--map-database", str(map_database),
                                "--map-database-mode", args.map_database_mode])
            if profile.get("command_speed_mps") is not None:
                command.extend(["--speed", str(profile["command_speed_mps"])])
            command.append("--gui" if gui else "--no-gui")
            if record:
                command.append("--record")
            if args.shutdown_gazebo and job_index == len(jobs) - 1:
                command.append("--shutdown-gazebo")
            update_state(state_path, "resetting", project=args.project, route_m=route)
            update_state(state_path, "running", project=args.project, route_m=route, command=command)
            environment = os.environ.copy()
            environment.update({
                "EXPERIMENT_PROJECT": args.project,
                "EXPERIMENT_SCENE": profile["scene"],
                "EXPERIMENT_ROUTE_M": str(route),
                "EXPERIMENT_GIT_COMMIT": manifest["git_commit"] or "unknown",
                "EXPERIMENT_CALIBRATION_VERSION": manifest["calibration_version"],
                "EXPERIMENT_SCENE_MANIFEST": str(scene_manifest_path) if scene_manifest_path else "",
                "EXPERIMENT_SCENE_MAP_VERSION": scene_manifest.get("map_version", "") if scene_manifest else "",
            })
            subprocess.run(command, cwd=root, env=environment, check=True)
            update_state(state_path, "collecting", project=args.project, route_m=route)
        if args.analyze or profile.get("analysis"):
            update_state(state_path, "analyzing", project=args.project)
            subprocess.run([sys.executable, str(analyzer), str(output),
                            "--route-m", str(max(route for route, _ in route_jobs(profile)))], cwd=root, check=True)
        if args.preflight:
            preflight = root / "src" / "visual_slam_gazebo" / "scripts" / "preflight_experiment_check.py"
            command = [sys.executable, str(preflight), "--repo", str(root),
                       "--status-root", str(output), "--output", str(output / "analysis" / "preflight")]
            if calibration_model:
                command.extend(["--calibration-model", str(calibration_model)])
            if args.dataset_manifest:
                command.extend(["--dataset-manifest", str(Path(args.dataset_manifest).resolve())])
            if args.separation_report:
                command.extend(["--separation-report", str(Path(args.separation_report).resolve())])
            subprocess.run(command, cwd=root, check=True)
        archive_path = None
        if args.archive:
            update_state(state_path, "archiving", project=args.project)
            archive_path = archive_campaign(root, args.project, output)
        manifest["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["map_database_sha256"] = (
            sha256(map_database) if map_database and map_database.exists() else None)
        manifest["archive"] = str(archive_path) if archive_path else None
        write_json(output / "campaign_manifest.json", manifest)
        update_state(state_path, "complete", project=args.project, output=str(output), archive=manifest["archive"])
        if archive_path:
            shutil.copy2(output / "campaign_manifest.json", archive_path / "manifest" / "campaign_manifest.json")
            shutil.copy2(state_path, archive_path / "manifest" / "campaign_state.json")
        print(json.dumps({"output": str(output), "archive": manifest["archive"]}, ensure_ascii=False))
    except KeyboardInterrupt:
        update_state(state_path, "interrupted", project=args.project, output=str(output))
        raise
    except Exception as exc:
        update_state(state_path, "failed", project=args.project, output=str(output), error=str(exc))
        raise


if __name__ == "__main__":
    if os.name == "nt":
        raise SystemExit(dispatch_windows_to_wsl(sys.argv[1:]))
    main()
