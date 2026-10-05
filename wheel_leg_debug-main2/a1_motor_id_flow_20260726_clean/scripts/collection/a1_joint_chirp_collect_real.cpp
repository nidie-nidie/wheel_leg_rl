#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "a1_unitree_lowlevel_common.hpp"

#include "a1_base/constants.hpp"

#include "unitree_legged_sdk/quadruped.h"
#include "unitree_legged_sdk/udp.h"
#if defined(__has_include)
#if __has_include("unitree_legged_sdk/safety.h")
#define A1_BASE_HAS_UNITREE_SAFETY_H 1
#include "unitree_legged_sdk/safety.h"
#else
#include "unitree_legged_sdk/control.h"
#endif
#else
#include "unitree_legged_sdk/control.h"
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <time.h>

namespace {

constexpr int kA1MotorCount = 12;
constexpr std::uint64_t kDtNs = 2000000ULL;
constexpr double kDtS = 0.002;
constexpr double kDefaultDurationS = 20.0;
constexpr double kDefaultF0Hz = 0.1;
constexpr double kDefaultF1Hz = 3.0;
constexpr double kDefaultKp = 20.0;
constexpr double kDefaultKd = 2.0;
constexpr double kDefaultDampingKd = 4.0;
constexpr double kDefaultHipCenterAbsRad = 0.05;
constexpr double kDefaultFrontThighCenterRad = 0.60;
constexpr double kDefaultRearThighCenterRad = 0.90;
constexpr double kDefaultCalfCenterRad = -1.55;
constexpr double kDefaultHipAmplitudeRad = 0.05;
constexpr double kDefaultThighAmplitudeRad = 0.10;
constexpr double kDefaultCalfAmplitudeRad = 0.10;
constexpr double kDefaultRampS = 2.0;
constexpr double kDefaultMaxAbsVelocityRadS = 20.0;
constexpr double kDefaultMaxAbsAccelerationRadS2 = 250.0;
constexpr double kDefaultLimitMarginRad = 0.08;
constexpr std::size_t kStartupReceiveAttempts = 2000;
constexpr std::size_t kShutdownDampingCycles = 250;
constexpr std::size_t kMaxConsecutiveMissedLowState = 50;
constexpr const char* kExecuteConfirmation = "A1_JOINT_CHIRP_COLLECT";

constexpr const char* kCsvHeader =
    "sample_index,t_monotonic_ns,recv_monotonic_ns,send_monotonic_ns,t_rel_s,stage,chirp_phase_rad,chirp_freq_hz,lowstate_tick,receive_status,send_status,kp,kd,"
    "q_des_FR_hip,q_des_FR_thigh,q_des_FR_calf,"
    "q_des_FL_hip,q_des_FL_thigh,q_des_FL_calf,"
    "q_des_RR_hip,q_des_RR_thigh,q_des_RR_calf,"
    "q_des_RL_hip,q_des_RL_thigh,q_des_RL_calf,"
    "q_FR_hip,q_FR_thigh,q_FR_calf,"
    "q_FL_hip,q_FL_thigh,q_FL_calf,"
    "q_RR_hip,q_RR_thigh,q_RR_calf,"
    "q_RL_hip,q_RL_thigh,q_RL_calf,"
    "dq_FR_hip,dq_FR_thigh,dq_FR_calf,"
    "dq_FL_hip,dq_FL_thigh,dq_FL_calf,"
    "dq_RR_hip,dq_RR_thigh,dq_RR_calf,"
    "dq_RL_hip,dq_RL_thigh,dq_RL_calf,"
    "tau_FR_hip,tau_FR_thigh,tau_FR_calf,"
    "tau_FL_hip,tau_FL_thigh,tau_FL_calf,"
    "tau_RR_hip,tau_RR_thigh,tau_RR_calf,"
    "tau_RL_hip,tau_RL_thigh,tau_RL_calf,"
    "temp_FR_hip,temp_FR_thigh,temp_FR_calf,"
    "temp_FL_hip,temp_FL_thigh,temp_FL_calf,"
    "temp_RR_hip,temp_RR_thigh,temp_RR_calf,"
    "temp_RL_hip,temp_RL_thigh,temp_RL_calf";

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_signal(int) { g_stop_requested = 1; }

typedef std::array<double, kA1MotorCount> JointValues;

#if defined(A1_BASE_HAS_UNITREE_SAFETY_H)
typedef UNITREE_LEGGED_SDK::Safety UnitreeSafetyAdapter;

UnitreeSafetyAdapter make_unitree_safety() {
  return UnitreeSafetyAdapter(UNITREE_LEGGED_SDK::LeggedType::A1);
}
#else
typedef UNITREE_LEGGED_SDK::Control UnitreeSafetyAdapter;

UnitreeSafetyAdapter make_unitree_safety() {
  return UnitreeSafetyAdapter(
      UNITREE_LEGGED_SDK::LeggedType::A1, UNITREE_LEGGED_SDK::LOWLEVEL);
}
#endif

std::unique_ptr<UNITREE_LEGGED_SDK::UDP> make_unitree_lowlevel_udp() {
  // Stock A1 SDK examples construct the UDP object with the default constructor
  // even for LOWLEVEL control. On that SDK, UDP(LOWLEVEL) resolves to the
  // high-level constructor path and leaves the send buffer at 143 bytes instead
  // of the 610-byte LowCmd packet expected by the motion controller.
  return std::unique_ptr<UNITREE_LEGGED_SDK::UDP>(
      new UNITREE_LEGGED_SDK::UDP());
}

void init_unitree_lowcmd(UnitreeSafetyAdapter& safety,
                         UNITREE_LEGGED_SDK::UDP& udp,
                         UNITREE_LEGGED_SDK::LowCmd& command) {
#if defined(A1_BASE_HAS_UNITREE_SAFETY_H)
  udp.InitCmdData(command);
#else
  safety.InitCmdData(command);
#endif
}

struct Options {
  bool help;
  bool describe_config;
  bool dry_run;
  bool execute;
  bool confirm_seen;
  std::string confirmation;
  std::string output_path;
  double duration_s;
  double f0_hz;
  double f1_hz;
  double kp;
  double kd;
  double damping_kd;
  double hip_center_abs_rad;
  double front_thigh_center_rad;
  double rear_thigh_center_rad;
  double calf_center_rad;
  double hip_amplitude_rad;
  double thigh_amplitude_rad;
  double calf_amplitude_rad;
  double ramp_s;
  double max_abs_velocity_rad_s;
  double max_abs_acceleration_rad_s2;
  double limit_margin_rad;

  Options()
      : help(false),
        describe_config(false),
        dry_run(false),
        execute(false),
        confirm_seen(false),
        duration_s(kDefaultDurationS),
        f0_hz(kDefaultF0Hz),
        f1_hz(kDefaultF1Hz),
        kp(kDefaultKp),
        kd(kDefaultKd),
        damping_kd(kDefaultDampingKd),
        hip_center_abs_rad(kDefaultHipCenterAbsRad),
        front_thigh_center_rad(kDefaultFrontThighCenterRad),
        rear_thigh_center_rad(kDefaultRearThighCenterRad),
        calf_center_rad(kDefaultCalfCenterRad),
        hip_amplitude_rad(kDefaultHipAmplitudeRad),
        thigh_amplitude_rad(kDefaultThighAmplitudeRad),
        calf_amplitude_rad(kDefaultCalfAmplitudeRad),
        ramp_s(kDefaultRampS),
        max_abs_velocity_rad_s(kDefaultMaxAbsVelocityRadS),
        max_abs_acceleration_rad_s2(kDefaultMaxAbsAccelerationRadS2),
        limit_margin_rad(kDefaultLimitMarginRad) {}
};

struct TargetSample {
  JointValues q_des;
  double phase_rad;
  double frequency_hz;

  TargetSample() : phase_rad(0.0), frequency_hz(0.0) {
    q_des.fill(0.0);
  }
};

struct MeasuredSample {
  JointValues q;
  JointValues dq;
  JointValues tau;
  JointValues temperature;
  std::uint32_t tick;

  MeasuredSample() : tick(0U) {
    q.fill(std::numeric_limits<double>::quiet_NaN());
    dq.fill(std::numeric_limits<double>::quiet_NaN());
    tau.fill(std::numeric_limits<double>::quiet_NaN());
    temperature.fill(std::numeric_limits<double>::quiet_NaN());
  }
};

int cli_error(const std::string& message) {
  std::cerr << "a1_joint_chirp_collect_real: " << message << std::endl;
  return 2;
}

void print_usage(std::ostream& stream) {
  stream
      << "Usage:\n"
      << "  a1_joint_chirp_collect_real --help\n"
      << "  a1_joint_chirp_collect_real --describe-config [OPTIONS]\n"
      << "  a1_joint_chirp_collect_real --dry-run --output CSV_PATH [OPTIONS]\n"
      << "  a1_joint_chirp_collect_real --execute --confirm A1_JOINT_CHIRP_COLLECT --output CSV_PATH [OPTIONS]\n"
      << "\n"
      << "Important options:\n"
      << "  --duration S                 Chirp duration. Default 20.0\n"
      << "  --f0 HZ                      Chirp start frequency. Default 0.1\n"
      << "  --f1 HZ                      Chirp end frequency. Default 3.0\n"
      << "  --kp VALUE                   Position Kp for all A1 joints. Default 20.0\n"
      << "  --kd VALUE                   Position Kd for all A1 joints. Default 2.0\n"
      << "  --damping-kd VALUE           Shutdown damping Kd. Default 4.0\n"
      << "  --hip-center-abs RAD         Left/right hip center magnitude. Default 0.05\n"
      << "  --front-thigh-center RAD     Front thigh center. Default 0.60\n"
      << "  --rear-thigh-center RAD      Rear thigh center. Default 0.90\n"
      << "  --calf-center RAD            Calf center. Default -1.55\n"
      << "  --hip-amp RAD                Hip chirp amplitude. Default 0.05\n"
      << "  --thigh-amp RAD              Thigh chirp amplitude. Default 0.10\n"
      << "  --calf-amp RAD               Calf chirp amplitude. Default 0.10\n"
      << "  --ramp S                     Chirp envelope ramp in/out. Default 2.0\n"
      << "  --max-vel RAD_PER_S          Preflight velocity limit. Default 20.0\n"
      << "  --max-accel RAD_PER_S2       Preflight acceleration limit. Default 250.0\n"
      << "  --limit-margin RAD           Joint limit margin. Default 0.08\n"
      << "\n"
      << "CSV joint order is Unitree raw order: FR, FL, RR, RL; hip, thigh, calf.\n";
}

bool parse_double(const char* text, double& value) {
  if (text == NULL || text[0] == '\0') return false;
  char* end = NULL;
  errno = 0;
  const double parsed = std::strtod(text, &end);
  if (errno != 0 || end == text || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

bool take_double_arg(int argc,
                     char** argv,
                     int& index,
                     const std::string& name,
                     double& destination,
                     std::string& error) {
  if (index + 1 >= argc) {
    error = name + " requires a numeric value";
    return false;
  }
  double parsed = 0.0;
  if (!parse_double(argv[index + 1], parsed)) {
    error = name + " requires a finite numeric value";
    return false;
  }
  destination = parsed;
  ++index;
  return true;
}

int parse_options(int argc, char** argv, Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help") {
      options.help = true;
      continue;
    }
    if (argument == "--describe-config") {
      options.describe_config = true;
      continue;
    }
    if (argument == "--dry-run") {
      options.dry_run = true;
      continue;
    }
    if (argument == "--execute") {
      options.execute = true;
      continue;
    }
    if (argument == "--confirm") {
      if (index + 1 >= argc) {
        return cli_error("--confirm requires a value");
      }
      options.confirm_seen = true;
      options.confirmation = argv[++index];
      continue;
    }
    if (argument == "--output") {
      if (index + 1 >= argc) {
        return cli_error("--output requires CSV_PATH");
      }
      options.output_path = argv[++index];
      continue;
    }

    std::string error;
    if (argument == "--duration") {
      if (!take_double_arg(argc, argv, index, argument, options.duration_s,
                           error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--f0") {
      if (!take_double_arg(argc, argv, index, argument, options.f0_hz,
                           error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--f1") {
      if (!take_double_arg(argc, argv, index, argument, options.f1_hz,
                           error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--kp") {
      if (!take_double_arg(argc, argv, index, argument, options.kp, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--kd") {
      if (!take_double_arg(argc, argv, index, argument, options.kd, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--damping-kd") {
      if (!take_double_arg(argc, argv, index, argument, options.damping_kd,
                           error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--hip-center-abs") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.hip_center_abs_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--front-thigh-center") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.front_thigh_center_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--rear-thigh-center") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.rear_thigh_center_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--calf-center") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.calf_center_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--hip-amp") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.hip_amplitude_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--thigh-amp") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.thigh_amplitude_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--calf-amp") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.calf_amplitude_rad, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--ramp") {
      if (!take_double_arg(argc, argv, index, argument, options.ramp_s,
                           error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--max-vel") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.max_abs_velocity_rad_s, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--max-accel") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.max_abs_acceleration_rad_s2, error)) {
        return cli_error(error);
      }
      continue;
    }
    if (argument == "--limit-margin") {
      if (!take_double_arg(argc, argv, index, argument,
                           options.limit_margin_rad, error)) {
        return cli_error(error);
      }
      continue;
    }

    return cli_error("unknown option: " + argument);
  }

  if (options.help) {
    return 0;
  }
  const int mode_count = (options.describe_config ? 1 : 0) +
                         (options.dry_run ? 1 : 0) +
                         (options.execute ? 1 : 0);
  if (mode_count != 1) {
    return cli_error("exactly one mode is required");
  }
  if (options.execute &&
      (!options.confirm_seen || options.confirmation != kExecuteConfirmation)) {
    return cli_error("--execute requires --confirm A1_JOINT_CHIRP_COLLECT");
  }
  if (!options.describe_config && options.output_path.empty()) {
    return cli_error("--dry-run and --execute require --output CSV_PATH");
  }
  return 0;
}

std::size_t sample_count(const Options& options) {
  const double exact = options.duration_s / kDtS;
  const double rounded = std::floor(exact + 0.5);
  if (!std::isfinite(exact) || rounded < 1.0 ||
      rounded >
          static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument("invalid sample count");
  }
  return static_cast<std::size_t>(rounded);
}

JointValues make_center(const Options& options) {
  JointValues center;
  center[0] = -options.hip_center_abs_rad;
  center[1] = options.front_thigh_center_rad;
  center[2] = options.calf_center_rad;
  center[3] = options.hip_center_abs_rad;
  center[4] = options.front_thigh_center_rad;
  center[5] = options.calf_center_rad;
  center[6] = -options.hip_center_abs_rad;
  center[7] = options.rear_thigh_center_rad;
  center[8] = options.calf_center_rad;
  center[9] = options.hip_center_abs_rad;
  center[10] = options.rear_thigh_center_rad;
  center[11] = options.calf_center_rad;
  return center;
}

JointValues make_amplitude(const Options& options) {
  JointValues amplitude;
  for (int leg = 0; leg < 4; ++leg) {
    const int base = 3 * leg;
    amplitude[base] = options.hip_amplitude_rad;
    amplitude[base + 1] = options.thigh_amplitude_rad;
    amplitude[base + 2] = options.calf_amplitude_rad;
  }
  return amplitude;
}

JointValues make_direction() {
  JointValues direction;
  direction[0] = -1.0;
  direction[1] = -1.0;
  direction[2] = -1.0;
  direction[3] = 1.0;
  direction[4] = 1.0;
  direction[5] = 1.0;
  direction[6] = -1.0;
  direction[7] = 1.0;
  direction[8] = 1.0;
  direction[9] = 1.0;
  direction[10] = -1.0;
  direction[11] = -1.0;
  return direction;
}

double smoothstep(double x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return x * x * (3.0 - 2.0 * x);
}

double chirp_envelope(const Options& options, double chirp_t_s) {
  if (options.ramp_s <= 0.0) return 1.0;
  const double effective_ramp =
      std::min(options.ramp_s, 0.5 * options.duration_s);
  if (effective_ramp <= 0.0) return 1.0;
  const double ramp_in = smoothstep(chirp_t_s / effective_ramp);
  const double ramp_out =
      smoothstep((options.duration_s - chirp_t_s) / effective_ramp);
  return std::min(ramp_in, ramp_out);
}

TargetSample make_chirp_target(const Options& options, double chirp_t_s) {
  TargetSample target;
  const JointValues center = make_center(options);
  const JointValues amplitude = make_amplitude(options);
  const JointValues direction = make_direction();
  const double beta_hz_s = (options.f1_hz - options.f0_hz) /
                           options.duration_s;
  target.frequency_hz = options.f0_hz + beta_hz_s * chirp_t_s;
  target.phase_rad = 2.0 * a1_base::constants::kPi *
                     (options.f0_hz * chirp_t_s +
                      0.5 * beta_hz_s * chirp_t_s * chirp_t_s);
  const double envelope = chirp_envelope(options, chirp_t_s);
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    target.q_des[motor] =
        center[motor] +
        direction[motor] * amplitude[motor] * envelope *
            std::sin(target.phase_rad);
  }
  return target;
}

JointValues interpolate(const JointValues& from,
                        const JointValues& to,
                        double alpha) {
  const double amount = smoothstep(alpha);
  JointValues result;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    result[motor] = from[motor] + amount * (to[motor] - from[motor]);
  }
  return result;
}

std::uint64_t monotonic_now_ns() {
  struct timespec now;
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
  }
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

struct timespec monotonic_timespec(std::uint64_t nanoseconds) {
  struct timespec value;
  value.tv_sec = static_cast<time_t>(nanoseconds / 1000000000ULL);
  value.tv_nsec = static_cast<long>(nanoseconds % 1000000000ULL);
  return value;
}

bool sleep_until_ns(std::uint64_t wake_ns) {
  const struct timespec wake_time = monotonic_timespec(wake_ns);
  int status = 0;
  do {
    status = ::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake_time, NULL);
  } while (status == EINTR && g_stop_requested == 0);
  return status == 0;
}

bool path_exists(const std::string& path, int& error_number) {
  struct stat information;
  if (::lstat(path.c_str(), &information) == 0) {
    error_number = 0;
    return true;
  }
  error_number = errno;
  return false;
}

int validate_output_path(const Options& options) {
  if (options.output_path.empty()) {
    return cli_error("--output requires CSV_PATH");
  }
  if (options.execute && options.output_path[0] != '/') {
    return cli_error("--execute output path must be absolute");
  }
  int path_error = 0;
  if (path_exists(options.output_path, path_error)) {
    return cli_error("output CSV already exists: " + options.output_path);
  }
  if (path_error != ENOENT) {
    return cli_error("cannot inspect output CSV: " +
                     std::string(std::strerror(path_error)));
  }
  return 0;
}

double lower_limit_for_motor(int motor) {
  switch (motor % 3) {
    case 0:
      return a1_base::constants::kHipMin;
    case 1:
      return a1_base::constants::kThighMin;
    default:
      return a1_base::constants::kCalfMin;
  }
}

double upper_limit_for_motor(int motor) {
  switch (motor % 3) {
    case 0:
      return a1_base::constants::kHipMax;
    case 1:
      return a1_base::constants::kThighMax;
    default:
      return a1_base::constants::kCalfMax;
  }
}

int validate_motion_config(const Options& options) {
  if (options.duration_s <= 0.0 || options.f0_hz <= 0.0 ||
      options.f1_hz < options.f0_hz) {
    return cli_error("invalid duration or chirp frequency range");
  }
  if (options.kp < 0.0 || options.kd < 0.0 || options.damping_kd < 0.0) {
    return cli_error("Kp/Kd must be non-negative");
  }
  if (options.hip_amplitude_rad < 0.0 ||
      options.thigh_amplitude_rad < 0.0 ||
      options.calf_amplitude_rad < 0.0 ||
      options.ramp_s < 0.0 ||
      options.max_abs_velocity_rad_s <= 0.0 ||
      options.max_abs_acceleration_rad_s2 <= 0.0 ||
      options.limit_margin_rad < 0.0) {
    return cli_error("invalid amplitude, ramp, or safety limit");
  }
  sample_count(options);

  const JointValues center = make_center(options);
  const JointValues amplitude = make_amplitude(options);
  const double beta_hz_s = (options.f1_hz - options.f0_hz) /
                           options.duration_s;
  const double omega_max = 2.0 * a1_base::constants::kPi * options.f1_hz;
  const double angular_acceleration =
      2.0 * a1_base::constants::kPi * beta_hz_s;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    const double lower = lower_limit_for_motor(motor) +
                         options.limit_margin_rad;
    const double upper = upper_limit_for_motor(motor) -
                         options.limit_margin_rad;
    const double min_q = center[motor] - amplitude[motor];
    const double max_q = center[motor] + amplitude[motor];
    if (min_q < lower || max_q > upper) {
      std::ostringstream message;
      message << "joint envelope too close to limit at motor " << motor
              << " min=" << min_q << " max=" << max_q
              << " allowed=[" << lower << "," << upper << "]";
      return cli_error(message.str());
    }
    const double peak_abs_velocity = amplitude[motor] * omega_max;
    const double peak_abs_acceleration =
        amplitude[motor] *
        (omega_max * omega_max + std::fabs(angular_acceleration));
    if (peak_abs_velocity > options.max_abs_velocity_rad_s) {
      std::ostringstream message;
      message << "joint velocity preflight failed at motor " << motor
              << " peak=" << peak_abs_velocity;
      return cli_error(message.str());
    }
    if (peak_abs_acceleration > options.max_abs_acceleration_rad_s2) {
      std::ostringstream message;
      message << "joint acceleration preflight failed at motor " << motor
              << " peak=" << peak_abs_acceleration;
      return cli_error(message.str());
    }
  }
  return 0;
}

void write_joint_values(std::ostream& csv, const JointValues& values) {
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    csv << ',' << values[motor];
  }
}

void write_csv_row(std::ostream& csv,
                   std::size_t sample_index,
                   std::uint64_t monotonic_ns,
                   std::uint64_t recv_monotonic_ns,
                   std::uint64_t send_monotonic_ns,
                   double relative_time_s,
                   const char* stage,
                   const TargetSample& target,
                   const MeasuredSample& measured,
                   int receive_status,
                   int send_status,
                   double kp,
                   double kd) {
  csv << sample_index << ',' << monotonic_ns << ',' << recv_monotonic_ns << ','
      << send_monotonic_ns << ',' << relative_time_s << ',' << stage << ','
      << target.phase_rad << ',' << target.frequency_hz << ','
      << measured.tick << ',' << receive_status << ',' << send_status << ','
      << kp << ',' << kd;
  write_joint_values(csv, target.q_des);
  write_joint_values(csv, measured.q);
  write_joint_values(csv, measured.dq);
  write_joint_values(csv, measured.tau);
  write_joint_values(csv, measured.temperature);
  csv << '\n';
}

MeasuredSample make_ideal_measurement(const TargetSample& target,
                                      std::uint32_t tick) {
  MeasuredSample measured;
  measured.tick = tick;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    measured.q[motor] = target.q_des[motor];
    measured.dq[motor] = 0.0;
    measured.tau[motor] = 0.0;
    measured.temperature[motor] = 35.0;
  }
  return measured;
}

MeasuredSample measured_from_lowstate(
    const UNITREE_LEGGED_SDK::LowState& sdk_state) {
  MeasuredSample measured;
  measured.tick = sdk_state.tick;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    measured.q[motor] = sdk_state.motorState[motor].q;
    measured.dq[motor] = sdk_state.motorState[motor].dq;
    measured.tau[motor] = sdk_state.motorState[motor].tauEst;
    measured.temperature[motor] = sdk_state.motorState[motor].temperature;
  }
  return measured;
}

void write_position_command(const JointValues& q_des,
                            double kp,
                            double kd,
                            UNITREE_LEGGED_SDK::LowCmd& sdk_command) {
  a1_base::A1LowCmd core_command;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    core_command.motor[motor].enabled = true;
    core_command.motor[motor].q_des = q_des[motor];
    core_command.motor[motor].dq_des = 0.0;
    core_command.motor[motor].kp = kp;
    core_command.motor[motor].kd = kd;
    core_command.motor[motor].tau_ff = 0.0;
  }
  a1_base::unitree::write_core_command(
      core_command, a1_base::PaceCommandMode::Position, sdk_command);
}

bool send_damping(UNITREE_LEGGED_SDK::UDP& udp,
                  UNITREE_LEGGED_SDK::LowCmd& sdk_command,
                  double damping_kd) {
  a1_base::unitree::write_damping_command(
      sdk_command, static_cast<float>(damping_kd));
  const int set_send_status = udp.SetSend(sdk_command);
  const int send_status = set_send_status == 0 ? udp.Send() : -1;
  return set_send_status == 0 && send_status >= 0;
}

bool validate_collection_motor_state(
    const UNITREE_LEGGED_SDK::LowState& sdk_state,
    std::string& reason) {
  const std::array<const char*, 12>& names =
      a1_base::unitree::unitree_joint_names();
  bool has_nonzero_motor_feedback = false;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    const UNITREE_LEGGED_SDK::MotorState& state = sdk_state.motorState[motor];
    if (!std::isfinite(static_cast<double>(state.q)) ||
        !std::isfinite(static_cast<double>(state.dq)) ||
        !std::isfinite(static_cast<double>(state.tauEst))) {
      reason = std::string("non-finite motor state at ") + names[motor];
      return false;
    }
    has_nonzero_motor_feedback =
        has_nonzero_motor_feedback ||
        std::fabs(static_cast<double>(state.q)) > 1.0e-6 ||
        std::fabs(static_cast<double>(state.dq)) > 1.0e-6 ||
        std::fabs(static_cast<double>(state.tauEst)) > 1.0e-6 ||
        state.temperature != 0 || state.mode != 0;
  }
  if (!has_nonzero_motor_feedback) {
    reason = "empty motor feedback packet";
    return false;
  }
  return true;
}

void send_shutdown_damping(UNITREE_LEGGED_SDK::UDP& udp,
                           UNITREE_LEGGED_SDK::LowCmd& sdk_command,
                           double damping_kd) {
  for (std::size_t cycle = 0; cycle < kShutdownDampingCycles; ++cycle) {
    send_damping(udp, sdk_command, damping_kd);
    const std::uint64_t wake_ns = monotonic_now_ns() + kDtNs;
    sleep_until_ns(wake_ns);
  }
}

bool receive_valid_lowstate(UNITREE_LEGGED_SDK::UDP& udp,
                            UNITREE_LEGGED_SDK::LowState& sdk_state,
                            int& receive_status,
                            std::string& reason) {
  receive_status = udp.Recv();
  udp.GetRecv(sdk_state);
  const bool state_valid =
      validate_collection_motor_state(sdk_state, reason);
  if (!state_valid && receive_status < 0) {
    reason = "invalid LowState after UDP receive status " +
             std::to_string(receive_status) + ": " + reason;
  }
  return state_valid;
}

int run_describe_config(const Options& options) {
  const JointValues center = make_center(options);
  const JointValues amplitude = make_amplitude(options);
  const JointValues direction = make_direction();
  std::cout << "{"
            << "\"schema_version\":\"a1_joint_chirp_collect_config/v1\","
            << "\"dt_s\":" << kDtS << ","
            << "\"duration_s\":" << options.duration_s << ","
            << "\"sample_count\":" << sample_count(options) << ","
            << "\"f0_hz\":" << options.f0_hz << ","
            << "\"f1_hz\":" << options.f1_hz << ","
            << "\"kp\":" << options.kp << ","
            << "\"kd\":" << options.kd << ","
            << "\"unitree_joint_order\":[";
  const std::array<const char*, 12>& names =
      a1_base::unitree::unitree_joint_names();
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    if (motor != 0) std::cout << ',';
    std::cout << '"' << names[motor] << '"';
  }
  std::cout << "],\"center_rad\":[";
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    if (motor != 0) std::cout << ',';
    std::cout << center[motor];
  }
  std::cout << "],\"amplitude_rad\":[";
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    if (motor != 0) std::cout << ',';
    std::cout << amplitude[motor];
  }
  std::cout << "],\"direction\":[";
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    if (motor != 0) std::cout << ',';
    std::cout << direction[motor];
  }
  std::cout << "]}" << std::endl;
  return 0;
}

int run_dry_run(const Options& options) {
  const int output_result = validate_output_path(options);
  if (output_result != 0) return output_result;

  std::ofstream csv(options.output_path.c_str(),
                    std::ios::out | std::ios::trunc);
  if (!csv) {
    return cli_error("cannot open output CSV: " + options.output_path);
  }
  csv << std::setprecision(9);
  csv << kCsvHeader << '\n';

  const std::size_t count = sample_count(options);
  const std::uint64_t start_ns = 0;
  for (std::size_t sample = 0; sample < count; ++sample) {
    const double t = static_cast<double>(sample) * kDtS;
    const TargetSample target = make_chirp_target(options, t);
    const MeasuredSample measured =
        make_ideal_measurement(target, static_cast<std::uint32_t>(sample + 1));
    write_csv_row(csv, sample, start_ns + sample * kDtNs,
                  start_ns + sample * kDtNs, start_ns + sample * kDtNs,
                  t, "chirp",
                  target, measured, 0, 610, options.kp, options.kd);
  }
  if (!csv) {
    return cli_error("failed while writing output CSV");
  }
  std::cout << "{\"status\":\"PASS\",\"mode\":\"dry-run\",\"samples\":"
            << count << ",\"output\":\"" << options.output_path << "\"}"
            << std::endl;
  return 0;
}

JointValues q_from_lowstate(const UNITREE_LEGGED_SDK::LowState& sdk_state) {
  JointValues q;
  for (int motor = 0; motor < kA1MotorCount; ++motor) {
    q[motor] = sdk_state.motorState[motor].q;
  }
  return q;
}

int run_execute(const Options& options) {
  const int output_result = validate_output_path(options);
  if (output_result != 0) return output_result;

  std::ofstream csv(options.output_path.c_str(),
                    std::ios::out | std::ios::trunc);
  if (!csv) {
    return cli_error("cannot open output CSV: " + options.output_path);
  }
  csv << std::setprecision(9);
  csv << kCsvHeader << '\n';

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  UnitreeSafetyAdapter safety = make_unitree_safety();
  std::unique_ptr<UNITREE_LEGGED_SDK::UDP> udp = make_unitree_lowlevel_udp();
  UNITREE_LEGGED_SDK::LowCmd sdk_command = {};
  UNITREE_LEGGED_SDK::LowState sdk_state = {};
  init_unitree_lowcmd(safety, *udp, sdk_command);
  const bool startup_prime_sent = send_damping(*udp, sdk_command, 0.0);
  (void)startup_prime_sent;

  std::string reason;
  int receive_status = -1;
  bool received = false;
  for (std::size_t attempt = 0; attempt < kStartupReceiveAttempts; ++attempt) {
    if (g_stop_requested != 0) {
      return cli_error("stop requested before startup LowState");
    }
    if (receive_valid_lowstate(*udp, sdk_state, receive_status, reason)) {
      received = true;
      break;
    }
    send_damping(*udp, sdk_command, 0.0);
    const std::uint64_t wake_ns = monotonic_now_ns() + kDtNs;
    sleep_until_ns(wake_ns);
  }
  if (!received) {
    return cli_error("cannot receive valid startup LowState: " + reason);
  }

  const JointValues initial_q = q_from_lowstate(sdk_state);
  const JointValues center_q = make_center(options);
  const std::size_t ramp_cycles =
      static_cast<std::size_t>(std::floor(options.ramp_s / kDtS + 0.5));
  const std::size_t chirp_cycles = sample_count(options);
  const std::size_t total_cycles = ramp_cycles + chirp_cycles;
  const std::uint64_t origin_ns = monotonic_now_ns() + 5 * kDtNs;
  std::size_t missed_lowstate = 0;
  std::size_t sample_index = 0;
  int last_send_status = 0;

  for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
    if (g_stop_requested != 0) break;
    const std::uint64_t scheduled_ns = origin_ns + cycle * kDtNs;
    if (!sleep_until_ns(scheduled_ns)) {
      g_stop_requested = 1;
      break;
    }

    std::string receive_reason;
    const bool state_valid =
        receive_valid_lowstate(*udp, sdk_state, receive_status, receive_reason);
    const std::uint64_t recv_monotonic_ns = monotonic_now_ns();
    if (!state_valid) {
      ++missed_lowstate;
    } else {
      missed_lowstate = 0;
    }
    if (missed_lowstate > kMaxConsecutiveMissedLowState) {
      g_stop_requested = 1;
      break;
    }

    TargetSample target;
    const char* stage = "chirp";
    if (cycle < ramp_cycles) {
      stage = "center_ramp";
      const double alpha =
          ramp_cycles == 0
              ? 1.0
              : static_cast<double>(cycle + 1) /
                    static_cast<double>(ramp_cycles);
      target.q_des = interpolate(initial_q, center_q, alpha);
      target.phase_rad = 0.0;
      target.frequency_hz = 0.0;
    } else {
      const double chirp_t =
          static_cast<double>(cycle - ramp_cycles) * kDtS;
      target = make_chirp_target(options, chirp_t);
    }

    write_position_command(target.q_des, options.kp, options.kd, sdk_command);
    safety.PositionLimit(sdk_command);
    safety.PowerProtect(sdk_command, sdk_state, 1);
    const int set_send_status = udp->SetSend(sdk_command);
    const int send_status = set_send_status == 0 ? udp->Send() : -1;
    const std::uint64_t send_monotonic_ns = monotonic_now_ns();
    last_send_status = send_status;

    const MeasuredSample measured =
        state_valid ? measured_from_lowstate(sdk_state) : MeasuredSample();
    const double relative_time_s =
        static_cast<double>(cycle) * kDtS;
    write_csv_row(csv, sample_index, monotonic_now_ns(), recv_monotonic_ns,
                  send_monotonic_ns, relative_time_s, stage, target, measured,
                  receive_status, send_status, options.kp, options.kd);
    ++sample_index;

    if (set_send_status != 0 || send_status < 0) {
      g_stop_requested = 1;
      break;
    }
  }

  send_shutdown_damping(*udp, sdk_command, options.damping_kd);
  if (!csv) {
    return cli_error("failed while writing output CSV");
  }
  const bool complete = sample_index == total_cycles && g_stop_requested == 0;
  std::cout << "{\"status\":\"" << (complete ? "PASS" : "FAIL")
            << "\",\"mode\":\"execute\",\"samples\":" << sample_index
            << ",\"last_send_status\":" << last_send_status
            << ",\"output\":\"" << options.output_path << "\"}"
            << std::endl;
  return complete ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options;
    const int parse_result = parse_options(argc, argv, options);
    if (parse_result != 0) return parse_result;
    if (options.help) {
      print_usage(std::cout);
      return 0;
    }
    const int validation = validate_motion_config(options);
    if (validation != 0) return validation;
    if (options.describe_config) return run_describe_config(options);
    if (options.dry_run) return run_dry_run(options);
    return run_execute(options);
  } catch (const std::exception& error) {
    return cli_error(std::string("unexpected exception: ") + error.what());
  } catch (...) {
    return cli_error("unexpected non-standard exception");
  }
}
