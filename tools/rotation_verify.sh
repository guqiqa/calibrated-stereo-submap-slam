#!/usr/bin/env bash
# Verify the short-window rotation commit policy at two slow turn rates.
cd /mnt/d/科研slam || exit 1
source /opt/ros/humble/setup.bash
source install/setup.bash
for rate in 0.05 0.1; do
  echo "[verify] rate="${rate}" rad/s"
  python3 tools/turn_suite.py --modes b2_dual --runs 1 --duration 400 \
    --max-turn-angular "${rate}" --timeout-min 14 \
    --root "experiments/rotation_commit_${rate}" --force
done
echo "[verify] done"
