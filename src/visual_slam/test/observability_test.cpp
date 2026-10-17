#undef NDEBUG

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "visual_slam/observability.hpp"

namespace
{

int failures = 0;
int checks = 0;

void expect_true(bool condition, const std::string & label)
{
  ++checks;
  if (!condition) {
    ++failures;
    std::cout << "FAIL: " << label << std::endl;
  }
}

void expect_eq(long long actual, long long expected, const std::string & label)
{
  ++checks;
  if (actual != expected) {
    ++failures;
    std::cout << "FAIL: " << label << " expected " << expected << " got " << actual << std::endl;
  }
}

using visual_slam::EvidenceVerdict;
using visual_slam::IntervalCloseReason;
using visual_slam::ObservabilityEvidence;
using visual_slam::ObservabilityMode;
using visual_slam::ObservabilityStateMachine;
using visual_slam::ObservabilityThresholds;
using visual_slam::VisualState;

ObservabilityEvidence good_evidence()
{
  ObservabilityEvidence evidence;
  evidence.available = true;
  evidence.features = 120;
  evidence.inliers = 90;
  evidence.inlier_ratio = 0.75;
  evidence.reprojection_px = 0.8;
  evidence.view_change_deg = 2.0;
  evidence.grid_coverage = 7;
  evidence.condition_number = 50.0;
  return evidence;
}

ObservabilityEvidence bad_evidence()
{
  ObservabilityEvidence evidence;
  evidence.available = true;
  evidence.features = 5;
  evidence.inliers = 2;
  evidence.inlier_ratio = 0.10;
  evidence.reprojection_px = 9.9;
  evidence.view_change_deg = 60.0;
  evidence.grid_coverage = 1;
  evidence.condition_number = 1.0e6;
  return evidence;
}

ObservabilityEvidence marginal_evidence()
{
  ObservabilityEvidence evidence;
  evidence.available = true;
  evidence.features = 20;
  evidence.inliers = 15;
  evidence.inlier_ratio = 0.35;
  evidence.reprojection_px = 2.5;
  evidence.view_change_deg = 17.0;
  evidence.grid_coverage = 4;
  evidence.condition_number = 5.0e3;
  return evidence;
}

ObservabilityEvidence unavailable_evidence()
{
  ObservabilityEvidence evidence;
  evidence.available = false;
  return evidence;
}

void test_thresholds_and_classification()
{
  ObservabilityThresholds thresholds;
  expect_true(thresholds.consistent(), "default thresholds are consistent");

  ObservabilityThresholds broken = thresholds;
  broken.min_features_ok = 5;
  broken.min_features_enter = 15;
  expect_true(!broken.consistent(), "inverted feature thresholds are rejected");

  expect_true(
    visual_slam::classify_evidence(good_evidence(), thresholds) == EvidenceVerdict::kQualified,
    "good evidence qualifies");
  expect_true(
    visual_slam::classify_evidence(marginal_evidence(), thresholds) == EvidenceVerdict::kMarginal,
    "mid band evidence is marginal");
  expect_true(
    visual_slam::classify_evidence(bad_evidence(), thresholds) == EvidenceVerdict::kDisqualified,
    "bad evidence is disqualified");
  expect_true(
    visual_slam::classify_evidence(unavailable_evidence(), thresholds) ==
    EvidenceVerdict::kDisqualified,
    "missing geometry is disqualified");

  const std::uint32_t flags = visual_slam::disqualifying_flags(bad_evidence(), thresholds);
  expect_true((flags & visual_slam::kEvidenceFewFeatures) != 0u, "few features flag set");
  expect_true((flags & visual_slam::kEvidenceLargeViewChange) != 0u, "large view change flag set");
  expect_true(
    visual_slam::flags_to_string(flags).find("few_features") != std::string::npos,
    "flag rendering contains the feature reason");
}

void test_unevaluated_metrics_are_neutral()
{
  ObservabilityThresholds thresholds;
  ObservabilityEvidence evidence;
  evidence.available = true;
  evidence.features = 80;
  evidence.inliers = 60;
  evidence.inlier_ratio = 0.7;
  evidence.reprojection_px = std::numeric_limits<double>::quiet_NaN();
  evidence.view_change_deg = std::numeric_limits<double>::infinity();
  evidence.grid_coverage = -1;
  evidence.condition_number = std::numeric_limits<double>::quiet_NaN();
  expect_true(
    visual_slam::classify_evidence(evidence, thresholds) == EvidenceVerdict::kQualified,
    "unevaluated optional metrics do not disqualify");

  ObservabilityEvidence still_bad;
  still_bad.available = true;
  still_bad.features = 5;
  still_bad.inliers = 2;
  still_bad.inlier_ratio = 0.1;
  still_bad.reprojection_px = std::numeric_limits<double>::quiet_NaN();
  still_bad.view_change_deg = std::numeric_limits<double>::quiet_NaN();
  still_bad.grid_coverage = -1;
  still_bad.condition_number = std::numeric_limits<double>::quiet_NaN();
  expect_true(
    visual_slam::classify_evidence(still_bad, thresholds) == EvidenceVerdict::kDisqualified,
    "measured evidence still disqualifies");
}

void test_measured_frame_ignores_distribution()
{
  ObservabilityThresholds thresholds;
  ObservabilityEvidence evidence = good_evidence();
  evidence.grid_coverage = 1;
  expect_true(
    visual_slam::classify_evidence(evidence, thresholds) == EvidenceVerdict::kDisqualified,
    "poor distribution disqualifies an unmeasured frame");
  evidence.qualified_measurement = true;
  expect_true(
    visual_slam::classify_evidence(evidence, thresholds) == EvidenceVerdict::kQualified,
    "poor distribution alone does not disqualify a measured frame");
}

void test_rotation_commit_requires_two_frames()
{
  using visual_slam::RotationCommitConfig;
  using visual_slam::RotationCommitTracker;
  RotationCommitTracker tracker{RotationCommitConfig{}};

  auto first = tracker.update(0.12, true);
  expect_true(first.candidate, "first frame is a candidate");
  expect_true(!first.commit, "a single frame never commits");
  expect_eq(first.pending_frames, 1, "one candidate frame pending");

  auto second = tracker.update(0.10, true);
  expect_true(second.commit, "two same-sign frames above the commit angle commit");
  expect_eq(second.pending_frames, 2, "two frames in the window");
  expect_eq(tracker.commit_frames(), 1, "one commit recorded");

  auto third = tracker.update(0.06, true);
  expect_true(!third.commit, "the window restarts after a commit");
  expect_eq(third.pending_frames, 1, "fresh window has one frame");

  tracker.reset();
  tracker.update(0.12, true);
  auto flipped = tracker.update(-0.09, true);
  expect_true(!flipped.commit, "a sign flip does not commit");
  expect_eq(flipped.sign, -1, "the new candidate takes the new sign");
  expect_eq(flipped.pending_frames, 1, "sign flip restarts the window");

  tracker.reset();
  tracker.update(0.02, true);
  auto weak = tracker.update(0.02, true);
  expect_true(!weak.candidate, "below the per-frame minimum is not a candidate");
  expect_true(weak.reason == "below_min_frame_deg", "weak frame reason recorded");

  tracker.reset();
  tracker.update(0.10, false);
  auto gated = tracker.update(0.10, true);
  expect_true(!gated.commit, "a frame that fails the quality gate never commits");
  expect_true(gated.candidate, "the next good frame starts a new candidate");

  tracker.reset();
  tracker.update(0.05, true);
  auto below = tracker.update(0.05, true);
  expect_true(!below.commit, "a window sum below the commit angle does not commit");
  tracker.update(0.05, true);
  auto reached = tracker.update(0.05, true);
  expect_true(reached.commit, "commit when the window sum reaches the commit angle");
  expect_eq(reached.pending_frames, 4, "four frames accumulated in the window");
}

void test_legacy_mode_never_suppresses()
{
  ObservabilityStateMachine machine(ObservabilityMode::kB0Plain, ObservabilityThresholds{});
  for (int i = 0; i < 10; ++i) {
    machine.update(bad_evidence());
  }
  expect_true(machine.state() == VisualState::kObserved, "B0 stays observed");
  expect_true(machine.measurement_valid(), "B0 always emits a measurement");
  expect_eq(static_cast<long long>(machine.intervals().size()), 0, "B0 records no interval");
  expect_eq(machine.total_unobserved_frames(), 0, "B0 counts no unobserved frames");
}

void test_entry_hysteresis_and_r_pre_freeze()
{
  ObservabilityThresholds thresholds;
  ObservabilityStateMachine machine(ObservabilityMode::kB2Dual, thresholds);

  machine.update(good_evidence());   // frame 0
  machine.update(good_evidence());   // frame 1
  expect_true(machine.state() == VisualState::kObserved, "starts observed");
  expect_eq(machine.r_pre_frame(), -1, "no R_pre before degradation");

  machine.update(bad_evidence());    // frame 2, first disqualified frame
  expect_true(machine.state() == VisualState::kObserved, "one bad frame does not switch state");

  machine.update(bad_evidence());    // frame 3, enters UNOBSERVED
  expect_true(machine.state() == VisualState::kUnobserved, "second consecutive bad frame enters UNOBSERVED");
  expect_true(!machine.measurement_valid(), "unobserved frames are not valid measurements");
  expect_eq(machine.r_pre_frame(), 1, "R_pre is the last qualified frame");
  expect_eq(static_cast<long long>(machine.intervals().size()), 1, "one interval opened");
  expect_eq(machine.current_interval().start_frame, 3, "interval starts on the entering frame");
  expect_eq(machine.current_interval().r_pre_frame, 1, "interval records R_pre");
  expect_true(!machine.current_interval().reacquired(), "interval is still open");
  expect_eq(machine.unresolved_interval_count(), 1, "open interval counts as unresolved");

  for (int i = 0; i < 20; ++i) {
    machine.update(bad_evidence());
    expect_eq(machine.r_pre_frame(), 1, "R_pre does not move while UNOBSERVED");
  }
  // The triggering frame already belongs to the interval, so the 20 extra
  // updates bring the total to 21.
  expect_eq(machine.total_unobserved_frames(), 21, "unobserved frames counted from the triggering frame");
  expect_eq(machine.measurements_suppressed(), 21, "suppressed measurements counted");
  expect_eq(machine.current_interval().frames, 21, "current interval frames stay in sync");
  expect_true(machine.frames_since_valid_measurement() > 20, "age grows while unobserved");
}

void test_marginal_breaks_runs()
{
  ObservabilityThresholds thresholds;
  ObservabilityStateMachine machine(ObservabilityMode::kB2Dual, thresholds);
  machine.update(good_evidence());
  machine.update(bad_evidence());
  machine.update(marginal_evidence());
  machine.update(bad_evidence());
  expect_true(
    machine.state() == VisualState::kObserved,
    "a marginal frame breaks the disqualification run");
}

void test_reacquisition_confirm_and_finalize()
{
  ObservabilityThresholds thresholds;
  ObservabilityStateMachine machine(ObservabilityMode::kB2Dual, thresholds);
  machine.update(good_evidence());
  machine.update(bad_evidence());
  machine.update(bad_evidence());    // frame 2 -> UNOBSERVED
  expect_true(machine.state() == VisualState::kUnobserved, "entered unobserved");

  machine.update(unavailable_evidence());   // frame 3, still unobserved
  machine.update(good_evidence());          // frame 4, qualified run 1
  machine.update(good_evidence());          // frame 5, qualified run 2
  expect_true(machine.state() == VisualState::kUnobserved, "two qualified frames are not enough to exit");

  machine.update(good_evidence());          // frame 6, qualified run 3 -> REACQUIRED
  expect_true(machine.state() == VisualState::kReacquired, "third qualified frame reacquires");
  expect_true(machine.measurement_valid(), "reacquired frames are valid measurements");
  expect_eq(machine.r_post_frame(), 6, "R_post is the reacquiring frame");
  expect_eq(machine.current_interval().end_frame, 6, "interval closes at the reacquiring frame");
  expect_true(machine.current_interval().reacquired(), "interval marked reacquired");
  expect_true(!machine.current_interval().finalized(), "interval waits for closure annotation");
  expect_eq(machine.unresolved_interval_count(), 1, "unfinalized interval is unresolved");

  expect_true(
    machine.finalize_interval(IntervalCloseReason::kAnchorValidation, 0.12, 1.5, 6),
    "interval finalized with closure geometry");
  expect_true(machine.current_interval().finalized(), "interval finalized");
  expect_eq(machine.unresolved_interval_count(), 0, "no unresolved interval remains");
  expect_true(
    machine.current_interval().closed_by == IntervalCloseReason::kAnchorValidation,
    "closure reason stored");
  expect_true(
    !machine.finalize_interval(IntervalCloseReason::kRelocalization, 0.0, 0.0, 6),
    "finalizing twice is rejected");

  machine.update(good_evidence());         // frame 7 -> back to OBSERVED
  expect_true(machine.state() == VisualState::kObserved, "REACQUIRED returns to OBSERVED");
}

void test_relocalization_timeout()
{
  ObservabilityThresholds thresholds;
  thresholds.relocalization_timeout_frames = 5;
  ObservabilityStateMachine machine(ObservabilityMode::kB2Dual, thresholds);
  machine.update(good_evidence());
  machine.update(bad_evidence());
  machine.update(bad_evidence());
  expect_true(!machine.relocalization_required(), "timeout not reached yet");
  for (int i = 0; i < 5; ++i) {
    machine.update(bad_evidence());
  }
  expect_true(machine.relocalization_required(), "timeout requests relocalization");
}

void test_b1_records_no_intervals()
{
  ObservabilityStateMachine machine(ObservabilityMode::kB1Flag, ObservabilityThresholds{});
  machine.update(good_evidence());
  machine.update(bad_evidence());
  machine.update(bad_evidence());
  expect_true(machine.state() == VisualState::kUnobserved, "B1 still flags degradation");
  expect_eq(static_cast<long long>(machine.intervals().size()), 0, "B1 records no interval");
  expect_eq(machine.total_unobserved_frames(), 0, "B1 does not run interval counters");
  expect_eq(machine.r_pre_frame(), 0, "B1 still exposes the last valid frame");
}

void test_reacquire_then_immediate_fallback()
{
  ObservabilityThresholds thresholds;
  ObservabilityStateMachine machine(ObservabilityMode::kB2Dual, thresholds);
  machine.update(good_evidence());
  machine.update(bad_evidence());
  machine.update(bad_evidence());
  machine.update(good_evidence());
  machine.update(good_evidence());
  machine.update(good_evidence());
  expect_true(machine.state() == VisualState::kReacquired, "first interval reacquired");
  machine.update(bad_evidence());
  machine.update(bad_evidence());
  expect_true(machine.state() == VisualState::kUnobserved, "fallback reopens unobserved");
  expect_eq(static_cast<long long>(machine.intervals().size()), 2, "second interval opened");
  expect_eq(machine.current_interval().id, 2, "interval ids increase");
}

void test_grid_coverage()
{
  struct Point
  {
    float x;
    float y;
  };
  const std::vector<Point> points{{10.0F, 10.0F}, {200.0F, 10.0F}, {10.0F, 200.0F},
    {200.0F, 200.0F}, {300.0F, 230.0F}};
  expect_eq(visual_slam::grid_coverage(points, 320.0, 240.0, 3), 5, "five grid cells covered");
  const std::vector<Point> outside{{-5.0F, 10.0F}, {400.0F, 10.0F}};
  expect_eq(visual_slam::grid_coverage(outside, 320.0, 240.0, 3), 0, "out of range points ignored");
  expect_eq(visual_slam::grid_coverage(points, 0.0, 240.0, 3), 0, "invalid size yields zero");
}

}  // namespace

int main()
{
  test_thresholds_and_classification();
  test_unevaluated_metrics_are_neutral();
  test_measured_frame_ignores_distribution();
  test_rotation_commit_requires_two_frames();
  test_legacy_mode_never_suppresses();
  test_entry_hysteresis_and_r_pre_freeze();
  test_marginal_breaks_runs();
  test_reacquisition_confirm_and_finalize();
  test_relocalization_timeout();
  test_b1_records_no_intervals();
  test_reacquire_then_immediate_fallback();
  test_grid_coverage();

  std::cout << "checks=" << checks << " failures=" << failures << std::endl;
  return failures == 0 ? 0 : 1;
}
