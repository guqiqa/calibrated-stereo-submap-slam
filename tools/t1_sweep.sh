#!/usr/bin/env bash
# T1 angular sweep: one headless run per turn rate, then attribution.
# Usage: bash tools/t1_sweep.sh [duration_s]
cd /mnt/d/科研slam || exit 1
source /opt/ros/humble/setup.bash
source install/setup.bash
duration="${1:-400}"
for rate in 0.05 0.1 0.2 0.4; do
  echo "[t1] rate="${rate}" rad/s duration="${duration}" s"
  python3 tools/turn_suite.py --modes b2_dual --runs 1 --duration "${duration}" \
    --max-turn-angular "${rate}" --timeout-min 14 \
    --root "experiments/t1_angular_${rate}" --force
done
echo "[t1] sweep done"
