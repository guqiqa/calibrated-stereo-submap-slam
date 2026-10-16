#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace visual_slam
{

// Visibility state of the current frame relative to the current trusted visual
// reference. See docs/视觉可观测性与双参考状态管理设计.md.
enum class VisualState
{
  kObserved = 0,
  kUnobserved = 1,
  kReacquired = 2,
};

// B0: legacy behaviour (always output a prediction, no bookkeeping).
// B1: single trusted reference plus a failure flag.
// B2: dual reference plus explicit unobserved interval bookkeeping.
enum class ObservabilityMode
{
  kB0Plain = 0,
  kB1Flag = 1,
  kB2Dual = 2,
};

enum class IntervalCloseReason
{
  kIncrementBridge = 0,
  kAnchorValidation = 1,
  kRelocalization = 2,
  kStillOpen = 3,
};

enum class EvidenceVerdict
{
  kQualified = 0,
  kMarginal = 1,
  kDisqualified = 2,
};

inline const char * to_string(VisualState value)
{
  switch (value) {
    case VisualState::kObserved: return "OBSERVED";
    case VisualState::kUnobserved: return "UNOBSERVED";
    case VisualState::kReacquired: return "REACQUIRED";
  }
  return "UNKNOWN";
}

inline const char * to_string(ObservabilityMode value)
{
  switch (value) {
    case ObservabilityMode::kB0Plain: return "b0_plain";
    case ObservabilityMode::kB1Flag: return "b1_flag";
    case ObservabilityMode::kB2Dual: return "b2_dual";
  }
  return "unknown";
}

inline const char * to_string(IntervalCloseReason value)
{
  switch (value) {
    case IntervalCloseReason::kIncrementBridge: return "increment_bridge";
    case IntervalCloseReason::kAnchorValidation: return "anchor_validation";
    case IntervalCloseReason::kRelocalization: return "relocalization";
    case IntervalCloseReason::kStillOpen: return "still_open";
  }
  return "unknown";
}

inline const char * to_string(EvidenceVerdict value)
{
  switch (value) {
    case EvidenceVerdict::kQualified: return "qualified";
    case EvidenceVerdict::kMarginal: return "marginal";
    case EvidenceVerdict::kDisqualified: return "disqualified";
  }
  return "unknown";
}

// Pre-registered observability thresholds. Structure is frozen; numbers are
// frozen after the T1 rotation sweep (see the design document, section 4).
struct ObservabilityThresholds
{
  bool enabled{true};
  int min_features_ok{30};
  int min_features_enter{15};
  int min_inliers_ok{20};
  int min_inliers_enter{10};
  double min_inlier_ratio_ok{0.45};
  double min_inlier_ratio_enter{0.25};
  double max_reprojection_ok{2.0};
  double max_reprojection_enter{3.0};
  double max_view_change_ok_deg{15.0};
  double max_view_change_enter_deg{20.0};
  int min_grid_coverage_ok{5};
  int min_grid_coverage_enter{3};
  double max_condition_ok{1.0e3};
  double max_condition_enter{1.0e4};
  int enter_frames{2};
  int exit_frames{3};
  int relocalization_timeout_frames{90};

  // The enter level must be strictly looser than the ok level, otherwise the
  // hysteresis band disappears and the state machine can oscillate.
  bool consistent() const
  {
    return min_features_ok >= min_features_enter &&
           min_inliers_ok >= min_inliers_enter &&
           min_inlier_ratio_ok >= min_inlier_ratio_enter &&
           max_reprojection_ok <= max_reprojection_enter &&
           max_view_change_ok_deg <= max_view_change_enter_deg &&
           min_grid_coverage_ok >= min_grid_coverage_enter &&
           max_condition_ok <= max_condition_enter &&
           enter_frames >= 1 && exit_frames >= 1 &&
           relocalization_timeout_frames >= 1;
  }
};

// Per-frame evidence, always measured against a candidate visual reference.
struct ObservabilityEvidence
{
  bool available{false};
  int features{0};
  int inliers{0};
  double inlier_ratio{0.0};
  double reprojection_px{std::numeric_limits<double>::infinity()};
  double view_change_deg{std::numeric_limits<double>::infinity()};
  int grid_coverage{0};
  double condition_number{std::numeric_limits<double>::infinity()};
};

enum : std::uint32_t
{
  kEvidenceUnavailable = 1u << 0,
  kEvidenceFewFeatures = 1u << 1,
  kEvidenceFewInliers = 1u << 2,
  kEvidenceLowInlierRatio = 1u << 3,
  kEvidenceHighReprojection = 1u << 4,
  kEvidenceLargeViewChange = 1u << 5,
  kEvidencePoorDistribution = 1u << 6,
  kEvidenceIllConditioned = 1u << 7,
};

// A non-finite double metric or a negative grid coverage means "not evaluated"
// and is skipped by both the qualification and the disqualification check. Only
// an explicitly measured value can fail a criterion.
inline bool evidence_evaluated(double value) { return std::isfinite(value); }

inline bool evidence_evaluated(int value) { return value >= 0; }

// Bit flags of every enter-level violation. Empty means fully qualified.
inline std::uint32_t disqualifying_flags(
  const ObservabilityEvidence & evidence, const ObservabilityThresholds & thresholds)
{
  std::uint32_t flags = 0;
  if (!evidence.available) {
    return kEvidenceUnavailable;
  }
  if (evidence.features < thresholds.min_features_enter) flags |= kEvidenceFewFeatures;
  if (evidence.inliers < thresholds.min_inliers_enter) flags |= kEvidenceFewInliers;
  if (evidence.inlier_ratio < thresholds.min_inlier_ratio_enter) flags |= kEvidenceLowInlierRatio;
  if (evidence_evaluated(evidence.reprojection_px) &&
    evidence.reprojection_px > thresholds.max_reprojection_enter)
  {
    flags |= kEvidenceHighReprojection;
  }
  if (evidence_evaluated(evidence.view_change_deg) &&
    evidence.view_change_deg > thresholds.max_view_change_enter_deg)
  {
    flags |= kEvidenceLargeViewChange;
  }
  if (evidence_evaluated(evidence.grid_coverage) &&
    evidence.grid_coverage < thresholds.min_grid_coverage_enter)
  {
    flags |= kEvidencePoorDistribution;
  }
  if (evidence_evaluated(evidence.condition_number) &&
    evidence.condition_number > thresholds.max_condition_enter)
  {
    flags |= kEvidenceIllConditioned;
  }
  return flags;
}

inline bool qualifies(
  const ObservabilityEvidence & evidence, const ObservabilityThresholds & thresholds)
{
  if (!evidence.available) {
    return false;
  }
  const bool reprojection_ok = !evidence_evaluated(evidence.reprojection_px) ||
    evidence.reprojection_px <= thresholds.max_reprojection_ok;
  const bool view_change_ok = !evidence_evaluated(evidence.view_change_deg) ||
    evidence.view_change_deg <= thresholds.max_view_change_ok_deg;
  const bool coverage_ok = !evidence_evaluated(evidence.grid_coverage) ||
    evidence.grid_coverage >= thresholds.min_grid_coverage_ok;
  const bool conditioning_ok = !evidence_evaluated(evidence.condition_number) ||
    evidence.condition_number <= thresholds.max_condition_ok;
  return evidence.features >= thresholds.min_features_ok &&
         evidence.inliers >= thresholds.min_inliers_ok &&
         evidence.inlier_ratio >= thresholds.min_inlier_ratio_ok &&
         reprojection_ok && view_change_ok && coverage_ok && conditioning_ok;
}

inline EvidenceVerdict classify_evidence(
  const ObservabilityEvidence & evidence, const ObservabilityThresholds & thresholds)
{
  if (disqualifying_flags(evidence, thresholds) != 0u) {
    return EvidenceVerdict::kDisqualified;
  }
  return qualifies(evidence, thresholds) ? EvidenceVerdict::kQualified : EvidenceVerdict::kMarginal;
}

inline std::string flags_to_string(std::uint32_t flags)
{
  if (flags == 0u) {
    return "none";
  }
  static const std::pair<std::uint32_t, const char *> names[] = {
    {kEvidenceUnavailable, "unavailable"},
    {kEvidenceFewFeatures, "few_features"},
    {kEvidenceFewInliers, "few_inliers"},
    {kEvidenceLowInlierRatio, "low_inlier_ratio"},
    {kEvidenceHighReprojection, "high_reprojection"},
    {kEvidenceLargeViewChange, "large_view_change"},
    {kEvidencePoorDistribution, "poor_distribution"},
    {kEvidenceIllConditioned, "ill_conditioned"},
  };
  std::string text;
  for (const auto & entry : names) {
    if ((flags & entry.first) == 0u) {
      continue;
    }
    if (!text.empty()) {
      text += '|';
    }
    text += entry.second;
  }
  return text;
}

// One explicitly recorded unobserved interval. Invariant I2: an interval is
// either finalized with a non-still_open reason or reported as unresolved.
struct UnobservedInterval
{
  int id{0};
  std::int64_t start_frame{0};
  std::int64_t end_frame{-1};
  std::int64_t r_pre_frame{-1};
  std::int64_t r_post_frame{-1};
  std::uint32_t enter_flags{0};
  ObservabilityEvidence enter_evidence{};
  IntervalCloseReason closed_by{IntervalCloseReason::kStillOpen};
  int frames{0};
  int measurements_suppressed{0};
  double closure_translation_m{std::numeric_limits<double>::quiet_NaN()};
  double closure_rotation_deg{std::numeric_limits<double>::quiet_NaN()};

  bool reacquired() const { return end_frame >= 0; }
  bool finalized() const
  {
    return reacquired() && closed_by != IntervalCloseReason::kStillOpen;
  }
};

// Pure C++17 observability state machine. It has no ROS or OpenCV dependency so
// that the transition logic can be unit tested and replayed offline.
class ObservabilityStateMachine
{
public:
  ObservabilityStateMachine() = default;

  ObservabilityStateMachine(ObservabilityMode mode, const ObservabilityThresholds & thresholds)
  : mode_(mode), thresholds_(thresholds)
  {
  }

  void reset()
  {
    state_ = VisualState::kObserved;
    last_verdict_ = EvidenceVerdict::kQualified;
    frame_index_ = 0;
    last_valid_frame_ = -1;
    r_pre_frame_ = -1;
    r_post_frame_ = -1;
    consecutive_qualified_ = 0;
    consecutive_disqualified_ = 0;
    unobserved_frames_ = 0;
    measurements_suppressed_ = 0;
    reacquire_confirm_frames_ = 0;
    current_interval_ = UnobservedInterval{};
    intervals_.clear();
    last_evidence_ = ObservabilityEvidence{};
  }

  ObservabilityMode mode() const { return mode_; }
  const ObservabilityThresholds & thresholds() const { return thresholds_; }
  VisualState state() const { return state_; }
  const char * state_name() const { return to_string(state_); }
  bool legacy_mode() const
  {
    return mode_ == ObservabilityMode::kB0Plain || !thresholds_.enabled;
  }
  bool records_intervals() const { return mode_ == ObservabilityMode::kB2Dual; }

  // Only OBSERVED and REACQUIRED frames may be used as visual measurements.
  bool measurement_valid() const
  {
    if (legacy_mode()) {
      return true;
    }
    return state_ == VisualState::kObserved || state_ == VisualState::kReacquired;
  }

  std::int64_t frame_index() const { return frame_index_; }
  std::int64_t last_valid_frame() const { return last_valid_frame_; }
  std::int64_t r_pre_frame() const { return r_pre_frame_; }
  std::int64_t r_post_frame() const { return r_post_frame_; }
  int consecutive_qualified_frames() const { return consecutive_qualified_; }
  int consecutive_disqualified_frames() const { return consecutive_disqualified_; }
  int reacquire_confirm_frames() const { return reacquire_confirm_frames_; }
  int total_unobserved_frames() const { return unobserved_frames_; }
  int measurements_suppressed() const { return measurements_suppressed_; }
  const ObservabilityEvidence & last_evidence() const { return last_evidence_; }
  EvidenceVerdict last_verdict() const { return last_verdict_; }
  const UnobservedInterval & current_interval() const { return current_interval_; }
  const std::vector<UnobservedInterval> & intervals() const { return intervals_; }

  std::int64_t frames_since_valid_measurement() const
  {
    if (last_valid_frame_ < 0) {
      return frame_index_;
    }
    return frame_index_ - last_valid_frame_;
  }

  bool relocalization_required() const
  {
    return state_ == VisualState::kUnobserved &&
           frames_since_valid_measurement() >= thresholds_.relocalization_timeout_frames;
  }

  int unresolved_interval_count() const
  {
    int count = 0;
    for (const auto & interval : intervals_) {
      if (!interval.finalized()) {
        ++count;
      }
    }
    return count;
  }

  // Advances the machine by one processed frame.
  void update(const ObservabilityEvidence & evidence)
  {
    last_evidence_ = evidence;
    if (legacy_mode()) {
      last_verdict_ = EvidenceVerdict::kQualified;
      state_ = VisualState::kObserved;
      r_pre_frame_ = -1;
      r_post_frame_ = -1;
      if (evidence.available) {
        last_valid_frame_ = frame_index_;
      }
      ++frame_index_;
      return;
    }

    const EvidenceVerdict verdict = classify_evidence(evidence, thresholds_);
    last_verdict_ = verdict;
    if (verdict == EvidenceVerdict::kQualified) {
      ++consecutive_qualified_;
      consecutive_disqualified_ = 0;
    } else if (verdict == EvidenceVerdict::kDisqualified) {
      ++consecutive_disqualified_;
      consecutive_qualified_ = 0;
    } else {
      // A marginal frame breaks both runs: entry and exit both require
      // consecutive evidence.
      consecutive_qualified_ = 0;
      consecutive_disqualified_ = 0;
    }

    switch (state_) {
      case VisualState::kObserved:
        if (verdict == EvidenceVerdict::kDisqualified &&
          consecutive_disqualified_ >= thresholds_.enter_frames)
        {
          enter_unobserved(evidence);
        }
        break;
      case VisualState::kUnobserved:
        ++unobserved_frames_;
        ++measurements_suppressed_;
        if (records_intervals() && !intervals_.empty()) {
          ++intervals_.back().frames;
          ++intervals_.back().measurements_suppressed;
        }
        if (verdict == EvidenceVerdict::kQualified &&
          consecutive_qualified_ >= thresholds_.exit_frames)
        {
          state_ = VisualState::kReacquired;
          reacquire_confirm_frames_ = consecutive_qualified_;
          r_post_frame_ = frame_index_;
          if (records_intervals() && !intervals_.empty()) {
            intervals_.back().end_frame = frame_index_;
            intervals_.back().r_post_frame = frame_index_;
            current_interval_ = intervals_.back();
          }
        }
        break;
      case VisualState::kReacquired:
        reacquire_confirm_frames_ = consecutive_qualified_;
        if (verdict == EvidenceVerdict::kDisqualified) {
          state_ = VisualState::kUnobserved;
          enter_unobserved(evidence);
        } else {
          state_ = VisualState::kObserved;
        }
        break;
    }

    if (verdict == EvidenceVerdict::kQualified && measurement_valid()) {
      last_valid_frame_ = frame_index_;
    }
    ++frame_index_;
  }

  // Called by the caller once it has computed the R_pre -> R_post closure from
  // its own geometry check. Returns false when there is nothing to finalize.
  bool finalize_interval(
    IntervalCloseReason reason, double translation_m, double rotation_deg,
    std::int64_t r_post_frame)
  {
    if (reason == IntervalCloseReason::kStillOpen) {
      return false;
    }
    for (auto interval = intervals_.rbegin(); interval != intervals_.rend(); ++interval) {
      if (interval->reacquired() && !interval->finalized()) {
        interval->closed_by = reason;
        interval->closure_translation_m = translation_m;
        interval->closure_rotation_deg = rotation_deg;
        if (r_post_frame >= 0) {
          interval->r_post_frame = r_post_frame;
        }
        if (current_interval_.id == interval->id) {
          current_interval_ = *interval;
        }
        return true;
      }
    }
    return false;
  }

private:
  void enter_unobserved(const ObservabilityEvidence & evidence)
  {
    state_ = VisualState::kUnobserved;
    consecutive_qualified_ = 0;
    const std::uint32_t flags = disqualifying_flags(evidence, thresholds_);
    if (records_intervals()) {
      UnobservedInterval interval;
      interval.id = static_cast<int>(intervals_.size()) + 1;
      interval.start_frame = frame_index_;
      // Invariant I1: R_pre is the last frame that produced a valid measurement
      // and never moves while the interval is open.
      interval.r_pre_frame = last_valid_frame_;
      r_pre_frame_ = last_valid_frame_;
      interval.enter_flags = flags;
      interval.enter_evidence = evidence;
      intervals_.push_back(interval);
      current_interval_ = intervals_.back();
    } else {
      r_pre_frame_ = last_valid_frame_;
    }
  }

  ObservabilityMode mode_{ObservabilityMode::kB0Plain};
  ObservabilityThresholds thresholds_{};
  VisualState state_{VisualState::kObserved};
  EvidenceVerdict last_verdict_{EvidenceVerdict::kQualified};
  std::int64_t frame_index_{0};
  std::int64_t last_valid_frame_{-1};
  std::int64_t r_pre_frame_{-1};
  std::int64_t r_post_frame_{-1};
  int consecutive_qualified_{0};
  int consecutive_disqualified_{0};
  int unobserved_frames_{0};
  int measurements_suppressed_{0};
  int reacquire_confirm_frames_{0};
  ObservabilityEvidence last_evidence_{};
  UnobservedInterval current_interval_{};
  std::vector<UnobservedInterval> intervals_{};
};

// Coverage of a 3x3 (or generic bins x bins) image grid by the given points.
// Kept template based so the header stays free of OpenCV.
template <typename PointT>
int grid_coverage(const std::vector<PointT> & points, double width, double height, int bins = 3)
{
  if (bins < 1 || !(width > 0.0) || !(height > 0.0)) {
    return 0;
  }
  std::vector<unsigned char> occupied(static_cast<std::size_t>(bins) * bins, 0);
  int count = 0;
  for (const auto & point : points) {
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    if (x < 0.0 || y < 0.0 || x >= width || y >= height) {
      continue;
    }
    const int column = std::min(bins - 1, static_cast<int>(x / width * bins));
    const int row = std::min(bins - 1, static_cast<int>(y / height * bins));
    const std::size_t index = static_cast<std::size_t>(row) * bins + column;
    if (occupied[index] == 0) {
      occupied[index] = 1;
      ++count;
    }
  }
  return count;
}

}  // namespace visual_slam
