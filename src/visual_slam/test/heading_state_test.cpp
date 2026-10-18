#undef NDEBUG

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "visual_slam/heading_state.hpp"

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

void expect_near(double actual, double expected, double tolerance, const std::string & label)
{
  ++checks;
  if (!(std::abs(actual - expected) <= tolerance)) {
    ++failures;
    std::cout << "FAIL: " << label << " expected " << expected << " got " << actual << std::endl;
  }
}

using visual_slam::HeadingCandidate;
using visual_slam::HeadingSource;
using visual_slam::HeadingState;

HeadingCandidate candidate(std::int64_t frame, double heading_deg, bool valid = true,
  HeadingSource source = HeadingSource::kMetricPnp)
{
  HeadingCandidate item;
  item.frame_index = frame;
  item.reference_frame = frame - 1;
  item.heading_deg = heading_deg;
  item.valid = valid;
  item.source = source;
  return item;
}

void test_single_frame_never_commits()
{
  HeadingState state;
  state.reset(0.0);
  const auto update = state.propose(candidate(1, 0.5));
  expect_true(update.accepted, "first candidate accepted");
  expect_true(!update.commit, "a single frame never commits");
  expect_near(state.committed_deg(), 0.0, 1e-9, "committed heading unchanged");
  expect_near(state.pending_delta_deg(), 0.5, 1e-9, "pending delta recorded");
}

void test_two_frames_commit_once()
{
  HeadingState state;
  state.reset(0.0);
  state.propose(candidate(1, 0.12));
  const auto second = state.propose(candidate(2, 0.24));
  expect_true(second.commit, "second same-sign frame commits");
  expect_near(state.committed_deg(), 0.24, 1e-9, "committed heading follows the candidate");
  expect_true(!state.pending(), "window reset after commit");
  expect_near(state.pending_delta_deg(), 0.0, 1e-9, "no residual pending delta");
  expect_true(state.commit_count() == 1, "one commit counted");
  const auto third = state.propose(candidate(3, 0.30));
  expect_true(!third.commit, "a fresh window starts after a commit");
}

void test_duplicate_frame_consumed_once()
{
  HeadingState state;
  state.reset(0.0);
  state.propose(candidate(7, 0.15));
  const auto duplicate = state.propose(candidate(7, 0.35));
  expect_true(!duplicate.commit, "a duplicate frame cannot commit");
  expect_true(duplicate.reason == "duplicate_frame", "duplicate reason recorded");
  expect_true(state.duplicate_count() == 1, "duplicate counted");
  expect_near(state.committed_deg(), 0.0, 1e-9, "duplicate does not move the heading");
  const auto next = state.propose(candidate(8, 0.36));
  expect_true(next.commit, "the next distinct frame can commit");
}

void test_rejected_candidates_do_not_pollute()
{
  HeadingState state;
  state.reset(10.0);
  state.propose(candidate(1, 10.15));
  const auto rejected = state.propose(candidate(2, 10.20, false));
  expect_true(!rejected.commit, "invalid candidate never commits");
  expect_true(rejected.reason == "quality_gate_failed", "quality reason recorded");
  expect_near(state.committed_deg(), 10.0, 1e-9, "rejected candidate leaves committed untouched");
  expect_true(!state.pending(), "rejected candidate resets the window");

  const auto below = state.propose(candidate(3, 10.02));
  expect_true(below.reason == "below_min_frame_deg", "sub-threshold candidate rejected");
  expect_near(state.committed_deg(), 10.0, 1e-9, "sub-threshold keeps committed");
}

void test_sign_flip_restarts_window()
{
  HeadingState state;
  state.reset(0.0);
  state.propose(candidate(1, 0.12));
  const auto flipped = state.propose(candidate(2, -0.12));
  expect_true(!flipped.commit, "sign flip does not commit");
  expect_true(flipped.reason == "candidate_started", "sign flip starts a new window");
  expect_true(state.pending_frames() == 1, "window restarted with one frame");
}

void test_wrap_across_pi()
{
  HeadingState state;
  state.reset(179.9);
  state.propose(candidate(1, -179.95));
  const auto second = state.propose(candidate(2, -179.70));
  expect_true(second.commit, "crossing the 180 degree branch still commits");
  expect_near(state.committed_deg(), -179.70, 1e-6, "committed heading wraps correctly");
}

void test_implausible_jump_rejected()
{
  HeadingState state;
  state.reset(0.0);
  const auto update = state.propose(candidate(1, 120.0));
  expect_true(!update.commit, "an implausible jump never commits");
  expect_true(update.reason == "implausible_jump", "jump reason recorded");
  expect_near(state.committed_deg(), 0.0, 1e-9, "committed heading unchanged");
}

void test_absolute_write_is_explicit()
{
  HeadingState state;
  state.reset(0.0);
  const auto update = state.set_absolute(90.0, HeadingSource::kRelocalization, 42,
    "relocalization_accepted");
  expect_true(update.absolute, "absolute update flagged");
  expect_true(update.commit, "absolute update commits immediately");
  expect_near(state.committed_deg(), 90.0, 1e-9, "absolute heading applied");
  expect_true(state.absolute_count() == 1, "absolute write counted");
  const auto next = state.propose(candidate(43, 90.10));
  expect_true(!next.commit, "the window restarts after an absolute write");
}

void test_metric_and_recovery_alternation()
{
  HeadingState state;
  state.reset(0.0);
  // metric frame, recovery frame, metric frame, recovery frame: one continuous
  // turn must commit once and never jump.
  state.propose(candidate(1, 0.10, true, HeadingSource::kMetricPnp));
  const auto commit = state.propose(candidate(2, 0.22, true, HeadingSource::kDescriptor));
  expect_true(commit.commit, "alternating sources confirm one window");
  expect_near(state.committed_deg(), 0.22, 1e-9, "heading follows the confirmed candidate");
  const auto next = state.propose(candidate(3, 0.31, true, HeadingSource::kShortBaseline));
  expect_true(!next.commit, "a fresh window after a commit needs its own confirmation");
  // Keep the values clear of the exact commit boundary (0.20 deg) so the test
  // checks the policy, not floating point rounding.
  const auto confirm = state.propose(candidate(4, 0.45, true, HeadingSource::kMetricPnp));
  expect_true(confirm.commit, "the next window commits across sources");
  expect_near(state.committed_deg(), 0.45, 1e-9, "heading follows the second confirmation");
  const auto after = state.propose(candidate(5, 0.45, true, HeadingSource::kMetricPnp));
  expect_true(!after.commit, "no autonomous drift after the commit");
}

void test_translation_uses_committed_heading()
{
  visual_slam::GroundPose pose;
  pose.x = 0.0;
  pose.y = 0.0;
  pose.heading_deg = 0.0;

  // A body-frame forward step at heading 0 goes along +x.
  auto straight = visual_slam::advance_ground_pose(pose, 0.0, 1.0, 0.0);
  expect_near(straight.x, 1.0, 1e-9, "forward step at zero heading");
  expect_near(straight.y, 0.0, 1e-9, "no lateral drift at zero heading");

  // The same step after a committed 90 degree turn goes along +y.
  auto turned = visual_slam::advance_ground_pose(pose, 90.0, 1.0, 0.0);
  expect_near(turned.x, 0.0, 1e-9, "turned step has no x component");
  expect_near(turned.y, 1.0, 1e-9, "turned step goes along +y");
  expect_near(turned.heading_deg, 90.0, 1e-9, "the committed heading is stored");

  // A pending candidate heading must not influence the translation.
  HeadingState state;
  state.reset(0.0);
  state.propose(candidate(1, 90.0, false));   // rejected candidate
  const auto after_reject = visual_slam::advance_ground_pose(pose, state.committed_deg(), 1.0, 0.0);
  expect_near(after_reject.y, 0.0, 1e-9, "a rejected candidate never rotates the translation");
  expect_near(after_reject.x, 1.0, 1e-9, "translation follows the committed heading only");
}

void test_turn_composition_order()
{
  // Turning first and then translating must differ from translating first.
  visual_slam::GroundPose pose;
  auto turn_then_move = visual_slam::advance_ground_pose(pose, 90.0, 1.0, 0.0);
  expect_near(turn_then_move.x, 0.0, 1e-9, "turn then move: x unchanged");
  expect_near(turn_then_move.y, 1.0, 1e-9, "turn then move: moved along +y");

  auto move_then_turn = visual_slam::advance_ground_pose(pose, 0.0, 1.0, 0.0);
  expect_near(move_then_turn.x, 1.0, 1e-9, "move then turn: moved along +x");
  expect_near(move_then_turn.y, 0.0, 1e-9, "move then turn: y unchanged");
  expect_true(std::abs(turn_then_move.x - move_then_turn.x) > 1e-6,
    "the composition order is observable and must stay explicit");
}

// Frame-level loop: the metric path proposes an absolute heading every frame,
// the recovery path offers a second observation for the same frame, and the
// position advances with the committed heading only.
void test_frame_loop_with_two_sources()
{
  HeadingState state;
  state.reset(0.0);
  visual_slam::GroundPose pose;
  double truth_deg = 0.0;
  const double rate_deg_per_frame = 0.10;
  double previous_committed = state.committed_deg();
  double max_commit_jump = 0.0;
  int duplicate_frames = 0;

  for (std::int64_t frame = 1; frame <= 200; ++frame) {
    truth_deg += rate_deg_per_frame;
    const auto metric = state.propose(candidate(frame, truth_deg + 0.01, true,
      HeadingSource::kMetricPnp));
    const auto recovery = state.propose(candidate(frame, truth_deg + 0.02, true,
      HeadingSource::kDescriptor));
    if (recovery.reason == "duplicate_frame") {
      ++duplicate_frames;
    }
    pose = visual_slam::advance_ground_pose(pose, state.committed_deg(), 0.01, 0.0);
    if (metric.commit) {
      max_commit_jump = std::max(max_commit_jump,
        std::abs(state.committed_deg() - previous_committed));
      previous_committed = state.committed_deg();
    }
  }

  expect_true(duplicate_frames == 200, "every second observation of a frame is a duplicate");
  expect_true(max_commit_jump <= 0.45, "no single commit moves the heading by more than one window");
  expect_true(state.committed_deg() <= truth_deg + 1e-6, "the heading never runs ahead of the truth");
  expect_near(state.committed_deg(), truth_deg, 0.45, "the heading tracks the truth within one window");
  expect_true(state.commit_count() > 1, "the loop actually committed");
  expect_near(pose.heading_deg, state.committed_deg(), 1e-9, "the pose carries the committed heading");
  expect_near(pose.x, 200 * 0.01, 0.2, "translation accumulated along the committed heading");
}

void test_rejected_frame_does_not_stall_the_window()
{
  HeadingState state;
  state.reset(0.0);
  state.propose(candidate(1, 0.12));
  state.propose(candidate(2, 0.24, false));   // burst of bad evidence
  const auto recovered = state.propose(candidate(3, 0.30));
  expect_true(recovered.accepted, "tracking resumes after a rejected frame");
  expect_true(!recovered.commit, "the window restarts after the rejection");
  const auto confirmed = state.propose(candidate(4, 0.42));
  expect_true(confirmed.commit, "the next same-sign frame confirms the window");
  expect_near(state.committed_deg(), 0.42, 1e-9, "the committed heading follows the confirmation");
}

void test_commit_ledger_conserves()
{
  HeadingState state;
  state.reset(10.0);
  state.propose(candidate(1, 10.12));
  state.propose(candidate(2, 10.25));
  state.propose(candidate(3, 10.50));
  state.propose(candidate(4, 10.72));
  state.propose(candidate(5, 10.95));

  expect_true(state.commit_count() >= 2, "several commits recorded");
  expect_true(state.conservation_ok(), "the ledger adds up to the committed heading");
  expect_near(state.cumulative_delta_deg(), state.committed_deg() - 10.0, 1.0e-6,
    "cumulative delta matches the net heading change");
  expect_true(!state.events().empty(), "events recorded");
  double sum = 0.0;
  for (const auto & event : state.events()) {
    sum += event.delta_deg;
    expect_true(event.frame_index > 0, "event carries a frame id");
    expect_true(event.source != visual_slam::HeadingSource::kUnknown, "event carries a source");
  }
  expect_near(sum, state.cumulative_delta_deg(), 1.0e-6, "event deltas sum to the cumulative delta");
}

void test_ledger_across_pi()
{
  HeadingState state;
  state.reset(179.9);
  state.propose(candidate(1, -179.95));
  state.propose(candidate(2, -179.70));
  expect_true(state.commit_count() >= 1, "commit across the branch");
  expect_true(state.conservation_ok(), "conservation holds across +-180");
  for (const auto & event : state.events()) {
    expect_true(std::abs(event.delta_deg) < 180.0, "no event delta is a full turn");
  }
}

void test_absolute_write_is_in_the_ledger()
{
  HeadingState state;
  state.reset(0.0);
  state.set_absolute(90.0, HeadingSource::kRelocalization, 7, "relocalization_accepted");
  expect_true(state.conservation_ok(), "absolute writes are booked too");
  expect_true(!state.events().empty(), "absolute write recorded as an event");
  expect_near(state.events().back().delta_deg, 90.0, 1.0e-6, "absolute event delta");
  expect_true(state.events().back().source == visual_slam::HeadingSource::kRelocalization,
    "absolute event keeps its source");
}

}  // namespace

int main()
{
  test_single_frame_never_commits();
  test_two_frames_commit_once();
  test_duplicate_frame_consumed_once();
  test_rejected_candidates_do_not_pollute();
  test_sign_flip_restarts_window();
  test_wrap_across_pi();
  test_implausible_jump_rejected();
  test_absolute_write_is_explicit();
  test_metric_and_recovery_alternation();
  test_translation_uses_committed_heading();
  test_turn_composition_order();
  test_frame_loop_with_two_sources();
  test_rejected_frame_does_not_stall_the_window();
  test_commit_ledger_conserves();
  test_ledger_across_pi();
  test_absolute_write_is_in_the_ledger();

  std::cout << "checks=" << checks << " failures=" << failures << std::endl;
  return failures == 0 ? 0 : 1;
}
