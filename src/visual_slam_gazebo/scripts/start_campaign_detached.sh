#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /mnt/d/科研slam/install/setup.bash

OUTPUT_ROOT="${1:-/mnt/d/科研slam/experiments/main_45m_gsp}"
PROFILE="${2:-main_45m}"
RUNS="${3:-10}"

mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${OUTPUT_ROOT}/campaign.log"
PID_FILE="${OUTPUT_ROOT}/campaign.pid"

if pgrep -f "[r]un_experiment_profile.py --profile ${PROFILE}" >/dev/null 2>&1; then
    echo "campaign already running for profile ${PROFILE}" >&2
    exit 2
fi

nohup setsid ros2 run visual_slam_gazebo run_experiment_profile.py \
    --profile "${PROFILE}" \
    --runs "${RUNS}" \
    --modes global fixed joint \
    --output-root "${OUTPUT_ROOT}" \
    >"${LOG_FILE}" 2>&1 < /dev/null &

campaign_pid=$!
printf '%s\n' "${campaign_pid}" >"${PID_FILE}"
echo "started_pid=${campaign_pid}"
