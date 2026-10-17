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

  std::cout << "checks=" << checks << " failures=" << failures << std::endl;
  return failures == 0 ? 0 : 1;
}
