#pragma once

#include "a1_base/pace_collector.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace a1_base {
namespace pace_collect {

constexpr std::size_t kBaselineFootSampleCount = 1000;
constexpr std::size_t kBaselineImuSampleCount = 1000;
constexpr std::uint64_t kFeedbackStartupTimeoutNs = 2000000000ULL;
constexpr std::uint64_t kOwnershipHeartbeatExpiryNs = 1000000000ULL;
constexpr std::size_t kOwnershipHeartbeatPollByteBudget = 64;
constexpr std::size_t kImuMotionConsecutiveSampleCount = 10;
typedef std::array<double, 4> FootForces;

struct OwnershipHeartbeatStatus {
  bool initial_heartbeat_received;
  bool ownership_fault;
  bool monitor_heartbeat_stale;
};

// The caller owns the descriptor and calls poll once per collection cycle.
class OwnershipHeartbeatGate {
 public:
  OwnershipHeartbeatGate(int monitor_fd, std::uint64_t start_ns);

  OwnershipHeartbeatStatus poll(std::uint64_t now_ns);

 private:
  OwnershipHeartbeatGate(const OwnershipHeartbeatGate&) = delete;
  OwnershipHeartbeatGate& operator=(const OwnershipHeartbeatGate&) = delete;

  int monitor_fd_;
  std::uint64_t last_heartbeat_ns_;
  bool initial_heartbeat_received_;
  bool ownership_fault_;
};

struct ImuSample {
  double gyro_norm;
  double roll;
  double pitch;
};

enum class ImuSignal : std::size_t {
  GyroNorm = 0,
  Roll = 1,
  Pitch = 2,
};

class BaselineImuDetector {
 public:
  BaselineImuDetector();

  bool record_baseline(const ImuSample& sample);
  bool finalize_baseline();
  bool detect_motion(const ImuSample& sample);

  double median(ImuSignal signal) const;
  double mad(ImuSignal signal) const;
  double threshold(ImuSignal signal) const;

 private:
  static constexpr std::size_t kSignalCount = 3;

  std::array<std::array<double, kBaselineImuSampleCount>, kSignalCount>
      samples_;
  std::array<double, kSignalCount> medians_;
  std::array<double, kSignalCount> mads_;
  std::array<double, kSignalCount> thresholds_;
  std::size_t consecutive_exceedances_;
  std::size_t sample_count_;
  bool finalized_;
};

class StartupTickLock {
 public:
  StartupTickLock();

  bool observe(std::uint32_t tick);
  bool observe_receive(int receive_status, bool state_valid,
                       std::uint32_t tick);
  bool ready() const;
  std::uint32_t tick_step() const;

 private:
  bool has_previous_;
  bool ready_;
  std::uint32_t previous_tick_;
  std::uint32_t candidate_step_;
  std::size_t matching_delta_count_;
};

class BaselineFootDetector {
 public:
  BaselineFootDetector();

  bool record_baseline(const FootForces& sample);
  bool finalize_baseline();
  bool detect_contact(const FootForces& sample);

  double median(std::size_t foot) const;
  double mad(std::size_t foot) const;
  double threshold(std::size_t foot) const;

 private:
  std::array<std::array<double, kBaselineFootSampleCount>, 4> samples_;
  FootForces medians_;
  FootForces mads_;
  FootForces thresholds_;
  std::array<std::size_t, 4> consecutive_exceedances_;
  std::size_t sample_count_;
  bool finalized_;
};

bool position_send_deadline_expired(const PaceSendPlan& plan,
                                    std::uint64_t final_pre_send_ns);
bool feedback_startup_timed_out(std::uint64_t start_ns,
                                std::uint64_t now_ns);
bool feedback_keepalive_required(PaceCollectorState state,
                                 bool pace_command_requested);
bool safety_send_requires_finish(const PaceOperationOutcome& outcome);
bool receive_call_completed(int receive_status);
bool send_call_completed(int set_send_status, int send_status);
bool next_control_wake_after_consumed_packet(
    std::uint64_t last_receive_ns, std::uint64_t now_ns,
    std::uint64_t packet_period_ns, std::uint64_t phase_offset_ns,
    std::uint64_t& wake_ns);
bool receive_poll_budget_remaining(std::uint64_t scheduled_wake_ns,
                                   std::uint64_t now_ns,
                                   std::uint64_t budget_ns);

}  // namespace pace_collect
}  // namespace a1_base
