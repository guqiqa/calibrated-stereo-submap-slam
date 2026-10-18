#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace visual_slam
{

// Wrap an angle in degrees into (-180, 180].
inline double wrap_deg_180(double degrees)
{
  double wrapped = std::fmod(degrees + 180.0, 360.0);
  if (wrapped < 0.0) {
    wrapped += 360.0;
  }
  return wrapped - 180.0;
}

enum class HeadingSource
{
  kUnknown = 0,
  kMetric2d = 1,
  kMetricPnp = 2,
  kMetricStereo3d = 3,
  kDescriptor = 4,
  kShortBaseline = 5,
  kInitialization = 6,
  kRelocalization = 7,
  kLoopClosure = 8,
};

inline const char * to_string(HeadingSource source)
{
  switch (source) {
    case HeadingSource::kUnknown: return "unknown";
    case HeadingSource::kMetric2d: return "metric_2d";
    case HeadingSource::kMetricPnp: return "metric_pnp";
    case HeadingSource::kMetricStereo3d: return "metric_stereo_3d";
    case HeadingSource::kDescriptor: return "descriptor";
    case HeadingSource::kShortBaseline: return "short_baseline";
    case HeadingSource::kInitialization: return "initialization";
    case HeadingSource::kRelocalization: return "relocalization";
    case HeadingSource::kLoopClosure: return "loop_closure";
  }
  return "unknown";
}

// One heading observation for one frame.
//
// The caller is responsible for converting its own estimate into an ABSOLUTE
// world heading (not a delta, not a body-frame rotation) before proposing it.
// The state class never guesses the source, the axis or the direction.
struct HeadingCandidate
{
  std::int64_t frame_index{0};
  std::int64_t reference_frame{-1};
  double heading_deg{0.0};
  bool valid{false};
  HeadingSource source{HeadingSource::kUnknown};
};

struct HeadingUpdate
{
  bool accepted{false};
  bool commit{false};
  bool absolute{false};
  std::string reason{"idle"};
  double delta_deg{0.0};
  double committed_deg{0.0};
};

// Single source of truth for the published heading.
//
// Invariants:
//  * one candidate per frame is consumed; duplicates are counted and ignored;
//  * a single frame never commits a heading change;
//  * rejected candidates reset the confirmation window and never touch the
//    committed heading;
//  * absolute writes (initialization / relocalization / loop closure) are
//    explicit and reported, never inferred.
class HeadingState
{
public:
  struct Config
  {
    double min_frame_deg{0.05};
    double commit_deg{0.20};
    int min_frames{2};
    double max_jump_deg{90.0};
  };

  HeadingState() = default;
  explicit HeadingState(const Config & config) : config_(config) {}

  void reset(double initial_heading_deg = 0.0)
  {
    committed_deg_ = wrap_deg_180(initial_heading_deg);
    pending_sign_ = 0;
    pending_frames_ = 0;
    pending_delta_deg_ = 0.0;
    last_frame_index_ = std::numeric_limits<std::int64_t>::min();
    commit_count_ = 0;
    reject_count_ = 0;
    duplicate_count_ = 0;
    absolute_count_ = 0;
  }

  const Config & config() const { return config_; }
  double committed_deg() const { return committed_deg_; }
  double pending_delta_deg() const { return pending_delta_deg_; }
  int pending_frames() const { return pending_frames_; }
  int commit_count() const { return commit_count_; }
  int reject_count() const { return reject_count_; }
  int duplicate_count() const { return duplicate_count_; }
  int absolute_count() const { return absolute_count_; }
  bool pending() const { return pending_frames_ > 0; }

  HeadingUpdate propose(const HeadingCandidate & candidate)
  {
    HeadingUpdate update;
    update.committed_deg = committed_deg_;
    if (candidate.frame_index == last_frame_index_) {
      ++duplicate_count_;
      update.reason = "duplicate_frame";
      return update;
    }
    last_frame_index_ = candidate.frame_index;

    if (!candidate.valid || !std::isfinite(candidate.heading_deg)) {
      reset_window();
      ++reject_count_;
      update.reason = "quality_gate_failed";
      return update;
    }
    const double delta = wrap_deg_180(candidate.heading_deg - committed_deg_);
    update.delta_deg = delta;
    if (std::abs(delta) > config_.max_jump_deg) {
      reset_window();
      ++reject_count_;
      update.reason = "implausible_jump";
      return update;
    }
    if (std::abs(delta) < config_.min_frame_deg) {
      reset_window();
      ++reject_count_;
      update.reason = "below_min_frame_deg";
      return update;
    }
    const int sign = delta > 0.0 ? 1 : -1;
    update.accepted = true;
    if (pending_frames_ == 0 || sign != pending_sign_) {
      pending_sign_ = sign;
      pending_frames_ = 1;
      pending_delta_deg_ = delta;
      update.reason = "candidate_started";
      return update;
    }
    ++pending_frames_;
    pending_delta_deg_ = delta;
    if (pending_frames_ >= std::max(2, config_.min_frames) &&
      std::abs(delta) >= config_.commit_deg)
    {
      committed_deg_ = wrap_deg_180(candidate.heading_deg);
      reset_window();
      ++commit_count_;
      update.commit = true;
      update.reason = "commit_confirmed";
      update.committed_deg = committed_deg_;
    } else {
      update.reason = "candidate_pending";
    }
    return update;
  }

  // Absolute writes bypass the confirmation window but are explicit.
  HeadingUpdate set_absolute(double heading_deg, HeadingSource source,
    std::int64_t frame_index, const std::string & reason)
  {
    HeadingUpdate update;
    update.absolute = true;
    update.reason = reason;
    update.delta_deg = wrap_deg_180(heading_deg - committed_deg_);
    if (!std::isfinite(heading_deg)) {
      ++reject_count_;
      update.reason = "absolute_non_finite";
      update.committed_deg = committed_deg_;
      return update;
    }
    (void)source;
    committed_deg_ = wrap_deg_180(heading_deg);
    last_frame_index_ = frame_index;
    reset_window();
    ++absolute_count_;
    update.accepted = true;
    update.commit = true;
    update.committed_deg = committed_deg_;
    return update;
  }

private:
  void reset_window()
  {
    pending_sign_ = 0;
    pending_frames_ = 0;
    pending_delta_deg_ = 0.0;
  }

  Config config_{};
  double committed_deg_{0.0};
  int pending_sign_{0};
  int pending_frames_{0};
  double pending_delta_deg_{0.0};
  std::int64_t last_frame_index_{std::numeric_limits<std::int64_t>::min()};
  int commit_count_{0};
  int reject_count_{0};
  int duplicate_count_{0};
  int absolute_count_{0};
};

// Ground-plane pose used by the decoupled update.
//
// Translation is expressed in the world frame and is composed with the
// COMMITTED heading, never with the pending candidate: a heading that has not
// been confirmed must not move the position.
struct GroundPose
{
  double x{0.0};
  double y{0.0};
  double heading_deg{0.0};
};

// Applies a body-frame translation increment (metres) using the committed
// heading, then stores the committed heading. The candidate heading is ignored
// by design.
inline GroundPose advance_ground_pose(
  const GroundPose & pose, double committed_heading_deg, double dx_body, double dy_body)
{
  const double heading_rad = committed_heading_deg * M_PI / 180.0;
  const double cosine = std::cos(heading_rad);
  const double sine = std::sin(heading_rad);
  GroundPose next;
  next.x = pose.x + cosine * dx_body - sine * dy_body;
  next.y = pose.y + sine * dx_body + cosine * dy_body;
  next.heading_deg = wrap_deg_180(committed_heading_deg);
  return next;
}

}  // namespace visual_slam
