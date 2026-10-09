#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /mnt/d/科研slam/install/setup.bash

ROOT="${1:-/mnt/d/科研slam/experiments/formal_highload_60_90_120m_20261009}"
mkdir -p "${ROOT}"
printf '%s\n' "$$" > "${ROOT}/campaign.pid"

for item in "highload_60m 60m" "highload_90m 90m" "highload_120m 120m"; do
    read -r profile label <<< "${item}"
    output="${ROOT}/${label}"
    mkdir -p "${output}"
    printf '%s START %s\n' "$(date --iso-8601=seconds)" "${profile}" >> "${ROOT}/campaign.log"
    ros2 run visual_slam_gazebo run_experiment_profile.py \
        --profile "${profile}" \
        --modes global fixed joint \
        --runs 10 \
        --no-gui \
        --shutdown-gazebo \
        --output-root "${output}" \
        > "${output}/campaign.log" 2>&1
    printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "${profile}" >> "${ROOT}/campaign.log"
done

printf '%s\n' complete > "${ROOT}/campaign_state.txt"
