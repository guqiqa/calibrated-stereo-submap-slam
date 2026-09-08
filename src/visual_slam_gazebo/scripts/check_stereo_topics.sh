#!/usr/bin/env bash
set -u

LEFT_IMAGE=/sim/left/image_raw
RIGHT_IMAGE=/sim/right/image_raw
LEFT_INFO=/sim/left/camera_info
RIGHT_INFO=/sim/right/camera_info

fail=0
check_topic() {
  local topic="$1"
  local type="$2"
  local actual
  actual=$(ros2 topic type "$topic" 2>/dev/null || true)
  if [[ "$actual" != "$type" ]]; then
    printf 'FAIL %-28s expected=%s actual=%s\n' "$topic" "$type" "${actual:-missing}"
    fail=1
  else
    printf 'PASS %-28s type=%s\n' "$topic" "$actual"
  fi
}

check_topic "$LEFT_IMAGE" sensor_msgs/msg/Image
check_topic "$RIGHT_IMAGE" sensor_msgs/msg/Image
check_topic "$LEFT_INFO" sensor_msgs/msg/CameraInfo
check_topic "$RIGHT_INFO" sensor_msgs/msg/CameraInfo

for topic in "$LEFT_IMAGE" "$RIGHT_IMAGE"; do
  sample=$(timeout 8s ros2 topic echo "$topic" --once 2>/dev/null || true)
  if grep -q 'height: 480' <<<"$sample" && grep -q 'width: 640' <<<"$sample"; then
    printf 'PASS %-28s resolution=640x480\n' "$topic"
  else
    printf 'FAIL %-28s resolution sample unavailable\n' "$topic"
    fail=1
  fi
done

for topic in "$LEFT_INFO" "$RIGHT_INFO"; do
  sample=$(timeout 8s ros2 topic echo "$topic" --once 2>/dev/null || true)
  if grep -q 'distortion_model: plumb_bob' <<<"$sample" && grep -q 'k:' <<<"$sample"; then
    printf 'PASS %-28s camera_info=valid\n' "$topic"
  else
    printf 'FAIL %-28s camera_info unavailable\n' "$topic"
    fail=1
  fi
done

tf_sample=$(timeout 8s ros2 topic echo /tf_static --once 2>/dev/null || true)
if grep -q 'stereo_left_optical_frame' <<<"$tf_sample" && grep -q 'stereo_right_optical_frame' <<<"$tf_sample"; then
  printf 'PASS %-28s stereo TF present\n' /tf_static
else
  printf 'FAIL %-28s stereo TF unavailable\n' /tf_static
  fail=1
fi

printf 'INFO image rate check: run ros2 topic hz %s and %s (target about 15 Hz)\n' "$LEFT_IMAGE" "$RIGHT_IMAGE"
exit "$fail"
