#include "a1_pace_collect_support.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <unistd.h>

namespace a1_base {
namespace pace_collect {
namespace {

constexpr double kImuGyroThresholdFloor = 0.15;
constexpr double kImuAngleThresholdFloor = 0.03490658503988659;

template <std::size_t Count>
double median_of(std::array<double, Count> values) {
  static_assert(Count > 0, "median requires at least one value");
  std::sort(values.begin(), values.end());
  const std::size_t upper = Count / 2;
  if ((Count % 2) != 0) return values[upper];
  return values[upper - 1] * 0.5 + values[upper] * 0.5;
}

std::size_t imu_signal_index(ImuSignal signal) {
  return static_cast<std::size_t>(signal);
}

bool imu_sample_is_finite(const ImuSample& sample) {
  return std::isfinite(sample.gyro_norm) && std::isfinite(sample.roll) &&
         std::isfinite(sample.pitch);
}

}  // namespace

OwnershipHeartbeatGate::OwnershipHeartbeatGate(int monitor_fd,
                                               std::uint64_t start_ns)
    : monitor_fd_(monitor_fd), last_heartbeat_ns_(start_ns),
      initial_heartbeat_received_(false), ownership_fault_(false) {
  if (monitor_fd_ < 0) {
    ownership_fault_ = true;
    return;
  }
  const int flags = ::fcntl(monitor_fd_, F_GETFL);
  if (flags < 0 || ::fcntl(monitor_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
    ownership_fault_ = true;
  }
}

OwnershipHeartbeatStatus OwnershipHeartbeatGate::poll(std::uint64_t now_ns) {
  if (!ownership_fault_) {
    std::size_t bytes_read = 0;
    while (bytes_read < kOwnershipHeartbeatPollByteBudget) {
      char protocol_byte = 0;
      const ssize_t read_count = ::read(monitor_fd_, &protocol_byte, 1);
      if (read_count == 1) {
        ++bytes_read;
        if (protocol_byte == 'H' ||
            (protocol_byte == 'W' && !initial_heartbeat_received_)) {
          if (protocol_byte == 'H') {
            initial_heartbeat_received_ = true;
          }
          if (now_ns >= last_heartbeat_ns_) {
            last_heartbeat_ns_ = now_ns;
          }
          continue;
        }
        ownership_fault_ = true;
        break;
      }
      if (read_count == 0) {
        ownership_fault_ = true;
        break;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      ownership_fault_ = true;
      break;
    }
    if (!ownership_fault_ &&
        bytes_read == kOwnershipHeartbeatPollByteBudget) {
      struct pollfd monitor_poll;
      monitor_poll.fd = monitor_fd_;
      monitor_poll.events = POLLIN;
      monitor_poll.revents = 0;
      if (::poll(&monitor_poll, 1, 0) != 0) {
        ownership_fault_ = true;
      }
    }
  }

  OwnershipHeartbeatStatus status;
  status.initial_heartbeat_received = initial_heartbeat_received_;
  status.ownership_fault = ownership_fault_;
  status.monitor_heartbeat_stale =
      now_ns >= last_heartbeat_ns_ &&
      now_ns - last_heartbeat_ns_ >= kOwnershipHeartbeatExpiryNs;
  return status;
}

BaselineImuDetector::BaselineImuDetector()
    : samples_(), medians_(), mads_(), thresholds_(),
      consecutive_exceedances_(0), sample_count_(0), finalized_(false) {}

bool BaselineImuDetector::record_baseline(const ImuSample& sample) {
  if (finalized_ || sample_count_ >= kBaselineImuSampleCount ||
      !imu_sample_is_finite(sample)) {
    return false;
  }
  samples_[imu_signal_index(ImuSignal::GyroNorm)][sample_count_] =
      sample.gyro_norm;
  samples_[imu_signal_index(ImuSignal::Roll)][sample_count_] = sample.roll;
  samples_[imu_signal_index(ImuSignal::Pitch)][sample_count_] = sample.pitch;
  ++sample_count_;
  return true;
}

bool BaselineImuDetector::finalize_baseline() {
  if (finalized_) return true;
  if (sample_count_ != kBaselineImuSampleCount) return false;

  std::array<double, kSignalCount> medians;
  std::array<double, kSignalCount> mads;
  std::array<double, kSignalCount> thresholds;
  for (std::size_t signal = 0; signal < kSignalCount; ++signal) {
    medians[signal] = median_of(samples_[signal]);
    std::array<double, kBaselineImuSampleCount> deviations;
    for (std::size_t index = 0; index < kBaselineImuSampleCount; ++index) {
      deviations[index] = std::fabs(samples_[signal][index] - medians[signal]);
    }
    mads[signal] = median_of(deviations);
    const double deviation_threshold = 10.0 * mads[signal];
    if (signal == imu_signal_index(ImuSignal::GyroNorm)) {
      thresholds[signal] = medians[signal] +
          std::max(kImuGyroThresholdFloor, deviation_threshold);
    } else {
      thresholds[signal] =
          std::max(kImuAngleThresholdFloor, deviation_threshold);
    }
    if (!std::isfinite(medians[signal]) || !std::isfinite(mads[signal]) ||
        !std::isfinite(thresholds[signal])) {
      return false;
    }
  }
  medians_ = medians;
  mads_ = mads;
  thresholds_ = thresholds;
  finalized_ = true;
  return true;
}

bool BaselineImuDetector::detect_motion(const ImuSample& sample) {
  if (!finalized_ || !imu_sample_is_finite(sample)) {
    consecutive_exceedances_ = 0;
    return true;
  }
  const bool exceeded =
      sample.gyro_norm > thresholds_[imu_signal_index(ImuSignal::GyroNorm)] ||
      std::fabs(sample.roll - medians_[imu_signal_index(ImuSignal::Roll)]) >
          thresholds_[imu_signal_index(ImuSignal::Roll)] ||
      std::fabs(sample.pitch - medians_[imu_signal_index(ImuSignal::Pitch)]) >
          thresholds_[imu_signal_index(ImuSignal::Pitch)];
  if (!exceeded) {
    consecutive_exceedances_ = 0;
    return false;
  }
  if (consecutive_exceedances_ < kImuMotionConsecutiveSampleCount) {
    ++consecutive_exceedances_;
  }
  return consecutive_exceedances_ >= kImuMotionConsecutiveSampleCount;
}

double BaselineImuDetector::median(ImuSignal signal) const {
  const std::size_t index = imu_signal_index(signal);
  assert(index < kSignalCount);
  return medians_[index];
}

double BaselineImuDetector::mad(ImuSignal signal) const {
  const std::size_t index = imu_signal_index(signal);
  assert(index < kSignalCount);
  return mads_[index];
}

double BaselineImuDetector::threshold(ImuSignal signal) const {
  const std::size_t index = imu_signal_index(signal);
  assert(index < kSignalCount);
  return thresholds_[index];
}

StartupTickLock::StartupTickLock()
    : has_previous_(false), ready_(false), previous_tick_(0),
      candidate_step_(0), matching_delta_count_(0) {}

bool StartupTickLock::observe(std::uint32_t tick) {
  if (ready_) return true;
  if (!has_previous_) {
    previous_tick_ = tick;
    has_previous_ = true;
    return false;
  }

  const std::uint32_t delta = tick - previous_tick_;
  previous_tick_ = tick;
  if (delta == 0U || delta >= 0x80000000U) {
    candidate_step_ = 0U;
    matching_delta_count_ = 0;
    return false;
  }
  if (matching_delta_count_ == 0 || delta != candidate_step_) {
    candidate_step_ = delta;
    matching_delta_count_ = 1;
  } else {
    ++matching_delta_count_;
  }
  ready_ = matching_delta_count_ == 7;
  return ready_;
}

bool StartupTickLock::observe_receive(int receive_status, bool state_valid,
                                      std::uint32_t tick) {
  if (receive_status != 0) return ready_;
  if (!state_valid) {
    *this = StartupTickLock();
    return false;
  }
  return observe(tick);
}

bool StartupTickLock::ready() const { return ready_; }

std::uint32_t StartupTickLock::tick_step() const {
  return ready_ ? candidate_step_ : 0U;
}

BaselineFootDetector::BaselineFootDetector()
    : samples_(), medians_(), mads_(), thresholds_(),
      consecutive_exceedances_(), sample_count_(0), finalized_(false) {}

bool BaselineFootDetector::record_baseline(const FootForces& sample) {
  if (finalized_ || sample_count_ >= kBaselineFootSampleCount) return false;
  for (std::size_t foot = 0; foot < 4; ++foot) {
    if (!std::isfinite(sample[foot])) return false;
  }
  for (std::size_t foot = 0; foot < 4; ++foot) {
    samples_[foot][sample_count_] = sample[foot];
  }
  ++sample_count_;
  return true;
}

bool BaselineFootDetector::finalize_baseline() {
  if (finalized_) return true;
  if (sample_count_ != kBaselineFootSampleCount) return false;
  for (std::size_t foot = 0; foot < 4; ++foot) {
    medians_[foot] = median_of(samples_[foot]);
    std::array<double, kBaselineFootSampleCount> deviations;
    for (std::size_t index = 0; index < kBaselineFootSampleCount; ++index) {
      deviations[index] =
          std::fabs(samples_[foot][index] - medians_[foot]);
    }
    mads_[foot] = median_of(deviations);
    thresholds_[foot] = std::max(10.0, 10.0 * mads_[foot]);
  }
  finalized_ = true;
  return true;
}

bool BaselineFootDetector::detect_contact(const FootForces& sample) {
  if (!finalized_) return false;
  bool contact = false;
  for (std::size_t foot = 0; foot < 4; ++foot) {
    const bool exceeded = !std::isfinite(sample[foot]) ||
        std::fabs(sample[foot] - medians_[foot]) > thresholds_[foot];
    if (exceeded) {
      if (consecutive_exceedances_[foot] < 3) {
        ++consecutive_exceedances_[foot];
      }
    } else {
      consecutive_exceedances_[foot] = 0;
    }
    contact = contact || consecutive_exceedances_[foot] >= 3;
  }
  return contact;
}

double BaselineFootDetector::median(std::size_t foot) const {
  assert(foot < 4);
  return medians_[foot];
}

double BaselineFootDetector::mad(std::size_t foot) const {
  assert(foot < 4);
  return mads_[foot];
}

double BaselineFootDetector::threshold(std::size_t foot) const {
  assert(foot < 4);
  return thresholds_[foot];
}

bool position_send_deadline_expired(const PaceSendPlan& plan,
                                    std::uint64_t final_pre_send_ns) {
  return plan.command_requested &&
         plan.command_mode == PaceCommandMode::Position &&
         final_pre_send_ns > plan.position_valid_until_monotonic_ns;
}

bool feedback_startup_timed_out(std::uint64_t start_ns,
                                std::uint64_t now_ns) {
  return now_ns >= start_ns &&
         now_ns - start_ns >= kFeedbackStartupTimeoutNs;
}

bool feedback_keepalive_required(PaceCollectorState state,
                                 bool pace_command_requested) {
  return state == PaceCollectorState::Baseline &&
         !pace_command_requested;
}

bool safety_send_requires_finish(const PaceOperationOutcome& outcome) {
  return outcome.safety_send_required() &&
         outcome.safety_send_plan.transaction_id != 0;
}

bool receive_call_completed(int receive_status) {
  return receive_status == 0;
}

bool send_call_completed(int set_send_status, int send_status) {
  return set_send_status == 0 && send_status > 0;
}

bool next_control_wake_after_consumed_packet(
    std::uint64_t last_receive_ns, std::uint64_t now_ns,
    std::uint64_t packet_period_ns, std::uint64_t phase_offset_ns,
    std::uint64_t& wake_ns) {
  const std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  if (packet_period_ns == 0 ||
      phase_offset_ns > maximum - packet_period_ns) {
    return false;
  }
  const std::uint64_t first_wake_delta =
      packet_period_ns + phase_offset_ns;
  if (last_receive_ns > maximum - first_wake_delta) return false;
  const std::uint64_t first_wake = last_receive_ns + first_wake_delta;
  if (first_wake > now_ns) {
    wake_ns = first_wake;
    return true;
  }

  const std::uint64_t remainder =
      (now_ns - first_wake) % packet_period_ns;
  const std::uint64_t distance = packet_period_ns - remainder;
  if (now_ns > maximum - distance) return false;
  wake_ns = now_ns + distance;
  return true;
}

bool receive_poll_budget_remaining(std::uint64_t scheduled_wake_ns,
                                   std::uint64_t now_ns,
                                   std::uint64_t budget_ns) {
  if (budget_ns == 0) return false;
  const std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t deadline = scheduled_wake_ns > maximum - budget_ns
      ? maximum
      : scheduled_wake_ns + budget_ns;
  return now_ns < deadline;
}

}  // namespace pace_collect
}  // namespace a1_base
