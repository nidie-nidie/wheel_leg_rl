#include "a1_pace_collection_io.hpp"

#include <json-c/json.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace a1_base {
namespace {

const char* const kUnitreeJointNames[12] = {
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
};

const char* const kPaceJointNames[12] = {
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
};

const int kPaceToUnitreeMotorIndices[12] = {
    3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8,
};

const std::size_t kExpectedRows = 15000;
const std::size_t kExpectedFitStart = 3000;
const std::size_t kExpectedFitEnd = 13000;
const std::size_t kExpectedFitCount = 10000;
const double kExpectedDtS = 0.002;

CollectionWriteResult failed_result(const std::string& error) {
  CollectionWriteResult result;
  result.success = false;
  result.csv_path.clear();
  result.status_path.clear();
  result.error = error;
  return result;
}

std::string join_path(const std::string& directory, const char* name) {
  if (!directory.empty() && directory[directory.size() - 1] == '/') {
    return directory + name;
  }
  return directory + "/" + name;
}

void write_csv_header(std::ostream& stream) {
  stream << "sample_index,collector_state,in_fit_window,recv_monotonic_ns,"
            "send_monotonic_ns,low_state_tick,power_v,power_a,"
            "imu_qw,imu_qx,imu_qy,imu_qz,imu_roll,imu_pitch,imu_yaw,"
            "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
            "foot_force_fr,foot_force_fl,foot_force_rr,foot_force_rl,"
            "foot_force_est_fr,foot_force_est_fl,foot_force_est_rr,"
            "foot_force_est_rl";
  const char* const command_fields[7] = {
      "excitation_q", "pre_protect_q", "sent_q", "kp", "kd", "dq_des",
      "tau_ff",
  };
  const char* const measured_fields[4] = {
      "q", "dq", "tau_est", "temperature_c",
  };
  for (std::size_t joint = 0; joint < 12; ++joint) {
    for (std::size_t field = 0; field < 7; ++field) {
      stream << ',' << kUnitreeJointNames[joint] << '_' << command_fields[field];
    }
    for (std::size_t field = 0; field < 4; ++field) {
      stream << ',' << kUnitreeJointNames[joint] << '_' << measured_fields[field];
    }
  }
  stream << ",stale_state,duplicate_tick,nonfinite_state,deadline_miss,"
            "buffer_overflow,joint_clamp,slew_clamp,power_protect_changed,"
            "foot_contact,imu_motion,ownership_fault,monitor_heartbeat_stale,"
            "operator_fixture_stop,fault_active,stop_reason";
  stream << '\n';
}

struct FileIdentity {
  FileIdentity() : device(0), inode(0), valid(false) {}

  dev_t device;
  ino_t inode;
  bool valid;
};

bool record_file_identity(int fd, FileIdentity& identity, std::string& error) {
  struct stat info;
  if (::fstat(fd, &info) != 0) {
    error = "cannot inspect temporary file: " +
            std::string(std::strerror(errno));
    return false;
  }
  if (!S_ISREG(info.st_mode)) {
    error = "temporary file is not a regular file";
    return false;
  }
  identity.device = info.st_dev;
  identity.inode = info.st_ino;
  identity.valid = true;
  return true;
}

bool create_temporary_file(int directory_fd,
                           const char* name,
                           const char* label,
                           int& fd,
                           FileIdentity& identity,
                           std::string& error) {
  fd = ::openat(directory_fd, name,
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
  if (fd < 0) {
    error = "cannot create " + std::string(label) + " temporary file: " +
            std::string(std::strerror(errno));
    return false;
  }
  if (!record_file_identity(fd, identity, error)) {
    ::close(fd);
    fd = -1;
    return false;
  }
  return true;
}

std::string fd_path(int fd) {
  std::ostringstream path;
  path << "/proc/self/fd/" << fd;
  return path.str();
}

bool write_csv_file(int directory_fd,
                    const char* name,
                    const PaceSampleBuffer& buffer,
                    FileIdentity& identity,
                    std::string& error) {
  int fd = -1;
  if (!create_temporary_file(directory_fd, name, "CSV", fd, identity,
                             error)) {
    return false;
  }
  std::ofstream stream(fd_path(fd).c_str(),
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    error = "cannot open CSV temporary file: " +
            std::string(std::strerror(errno));
    ::close(fd);
    return false;
  }
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  write_csv_header(stream);
  for (std::size_t index = 0; index < buffer.size(); ++index) {
    const PaceRawSample& sample = buffer[index];
    stream << sample.sample_index << ','
           << pace_collector_state_name(sample.collector_state) << ','
           << (sample.in_fit_window ? 1 : 0) << ','
           << sample.recv_monotonic_ns << ','
           << (sample.command_sent ? std::to_string(sample.send_monotonic_ns)
                                   : std::string()) << ','
           << sample.measured_state.tick << ",,,";
    for (std::size_t component = 0; component < 4; ++component) {
      stream << sample.measured_state.imu.quaternion[component] << ',';
    }
    for (std::size_t component = 0; component < 3; ++component) {
      stream << sample.measured_state.imu.rpy[component] << ',';
    }
    for (std::size_t component = 0; component < 3; ++component) {
      stream << sample.measured_state.imu.gyroscope[component] << ',';
    }
    for (std::size_t component = 0; component < 3; ++component) {
      stream << sample.measured_state.imu.accelerometer[component] << ',';
    }
    for (std::size_t foot = 0; foot < 4; ++foot) {
      stream << sample.measured_state.foot_force[foot] << ',';
    }
    for (std::size_t foot = 0; foot < 4; ++foot) {
      stream << sample.measured_state.foot_force_est[foot];
      if (foot + 1 != 4) stream << ',';
    }
    for (std::size_t joint = 0; joint < 12; ++joint) {
      const A1MotorState& motor = sample.measured_state.motor[joint];
      stream << ',';
      if (sample.command_sent) {
        const A1MotorCmd& pre_protect = sample.pre_protect_command.motor[joint];
        const A1MotorCmd& sent = sample.sent_command.motor[joint];
        stream << sample.excitation_q.values[joint] << ','
               << pre_protect.q_des << ','
               << sent.q_des << ','
               << sent.kp << ','
               << sent.kd << ','
               << sent.dq_des << ','
               << sent.tau_ff;
      } else {
        stream << ",,,,,,";
      }
      stream << ',' << motor.q
             << ',' << motor.dq
             << ',' << motor.tau_est
             << ',' << motor.temperature;
    }
    stream << ',' << (sample.flags.stale_state ? 1 : 0)
           << ',' << (sample.flags.duplicate_tick ? 1 : 0)
           << ',' << (sample.flags.nonfinite_state ? 1 : 0)
           << ',' << (sample.flags.deadline_miss ? 1 : 0)
           << ',' << (sample.flags.buffer_overflow ? 1 : 0)
           << ',' << (sample.flags.joint_clamp ? 1 : 0)
           << ',' << (sample.flags.slew_clamp ? 1 : 0)
           << ',' << (sample.flags.power_protect_changed ? 1 : 0)
           << ',' << (sample.flags.foot_contact ? 1 : 0)
           << ',' << (sample.flags.imu_motion ? 1 : 0)
           << ',' << (sample.flags.ownership_fault ? 1 : 0)
           << ',' << (sample.flags.monitor_heartbeat_stale ? 1 : 0)
           << ',' << (sample.flags.operator_fixture_stop ? 1 : 0)
           << ',' << (sample.flags.fault_active ? 1 : 0)
           << ',';
    if (sample.stop_reason != PaceStopReason::None) {
      stream << pace_stop_reason_name(sample.stop_reason);
    }
    stream << '\n';
    if (!stream.good()) {
      error = "cannot write CSV temporary file";
      stream.close();
      ::close(fd);
      return false;
    }
  }
  stream.flush();
  if (!stream.good()) {
    error = "cannot flush CSV temporary file";
    stream.close();
    ::close(fd);
    return false;
  }
  stream.close();
  if (stream.fail()) {
    error = "cannot close CSV temporary file";
    ::close(fd);
    return false;
  }
  if (::close(fd) != 0) {
    error = "cannot close CSV temporary file";
    return false;
  }
  return true;
}

json_object* make_string_array(const char* const values[12]) {
  json_object* array = json_object_new_array();
  if (array == NULL) return NULL;
  for (std::size_t index = 0; index < 12; ++index) {
    json_object* value = json_object_new_string(values[index]);
    if (value == NULL) {
      json_object_put(array);
      return NULL;
    }
    json_object_array_add(array, value);
  }
  return array;
}

json_object* make_mapping_array() {
  json_object* array = json_object_new_array();
  if (array == NULL) return NULL;
  for (std::size_t index = 0; index < 12; ++index) {
    json_object* value =
        json_object_new_int(kPaceToUnitreeMotorIndices[index]);
    if (value == NULL) {
      json_object_put(array);
      return NULL;
    }
    json_object_array_add(array, value);
  }
  return array;
}

bool make_status_json(const CollectionWriteRequest& request,
                      const PaceSampleBuffer& buffer,
                      const PaceCollector& collector,
                      std::string& text,
                      std::string& error) {
  const bool pass =
      collector.state() == PaceCollectorState::Complete &&
      collector.trial_valid() && request.trial_valid &&
      request.stop_reason == PaceStopReason::None &&
      collector.stop_reason() == PaceStopReason::None &&
      collector.dropped_sample_count() == 0 &&
      buffer.size() == kExpectedRows &&
      collector.fit_window_start() == kExpectedFitStart &&
      collector.fit_window_end() == kExpectedFitEnd &&
      collector.fit_window_count() == kExpectedFitCount;

  json_object* root = json_object_new_object();
  json_object* unitree_names = make_string_array(kUnitreeJointNames);
  json_object* pace_names = make_string_array(kPaceJointNames);
  json_object* mapping = make_mapping_array();
  if (root == NULL || unitree_names == NULL || pace_names == NULL ||
      mapping == NULL) {
    if (root != NULL) json_object_put(root);
    if (unitree_names != NULL) json_object_put(unitree_names);
    if (pace_names != NULL) json_object_put(pace_names);
    if (mapping != NULL) json_object_put(mapping);
    error = "cannot allocate status JSON";
    return false;
  }

  json_object_object_add(
      root, "schema_version",
      json_object_new_string("a1_pace_collection_status/v1"));
  json_object_object_add(root, "collection_mode",
                         json_object_new_string(request.collection_mode.c_str()));
  json_object_object_add(root, "status",
                         json_object_new_string(pass ? "PASS" : "FAIL"));
  json_object_object_add(root, "trial_valid",
                         json_object_new_boolean(request.trial_valid));
  json_object_object_add(
      root, "stop_reason",
      json_object_new_string(pace_stop_reason_name(request.stop_reason)));
  json_object_object_add(
      root, "row_count",
      json_object_new_int64(static_cast<std::int64_t>(buffer.size())));
  json_object_object_add(
      root, "fit_window_start",
      json_object_new_int64(
          static_cast<std::int64_t>(collector.fit_window_start())));
  json_object_object_add(
      root, "fit_window_end",
      json_object_new_int64(
          static_cast<std::int64_t>(collector.fit_window_end())));
  json_object_object_add(
      root, "fit_window_count",
      json_object_new_int64(
          static_cast<std::int64_t>(collector.fit_window_count())));
  json_object_object_add(root, "dt_s", json_object_new_double(request.dt_s));
  json_object_object_add(
      root, "low_state_tick_step",
      json_object_new_int64(
          static_cast<std::int64_t>(request.low_state_tick_step)));
  json_object_object_add(
      root, "collector_state",
      json_object_new_string(pace_collector_state_name(collector.state())));
  json_object_object_add(
      root, "dropped_sample_count",
      json_object_new_int64(
          static_cast<std::int64_t>(collector.dropped_sample_count())));
  json_object_object_add(root, "unitree_joint_order", unitree_names);
  json_object_object_add(root, "pace_joint_order", pace_names);
  json_object_object_add(root, "pace_to_unitree_motor_indices", mapping);

  const char* serialized =
      json_object_to_json_string_ext(root, JSON_C_TO_STRING_PLAIN);
  if (serialized == NULL) {
    json_object_put(root);
    error = "cannot serialize status JSON";
    return false;
  }
  text = serialized;
  text.push_back('\n');
  json_object_put(root);
  return true;
}

bool write_text_file(int directory_fd,
                     const char* name,
                     const std::string& text,
                     FileIdentity& identity,
                     std::string& error) {
  int fd = -1;
  if (!create_temporary_file(directory_fd, name, "status", fd, identity,
                             error)) {
    return false;
  }
  std::ofstream stream(fd_path(fd).c_str(),
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    error = "cannot open status temporary file: " +
            std::string(std::strerror(errno));
    ::close(fd);
    return false;
  }
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  stream.flush();
  if (!stream.good()) {
    error = "cannot write status temporary file";
    stream.close();
    ::close(fd);
    return false;
  }
  stream.close();
  if (stream.fail()) {
    error = "cannot close status temporary file";
    ::close(fd);
    return false;
  }
  if (::close(fd) != 0) {
    error = "cannot close status temporary file";
    return false;
  }
  return true;
}

bool directory_is_empty(int directory_fd, std::string& error) {
  const int scan_fd = ::dup(directory_fd);
  if (scan_fd < 0) {
    error = "cannot duplicate reserved output directory: " +
            std::string(std::strerror(errno));
    return false;
  }
  DIR* stream = ::fdopendir(scan_fd);
  if (stream == NULL) {
    ::close(scan_fd);
    error = "cannot open reserved output directory: " +
            std::string(std::strerror(errno));
    return false;
  }
  bool empty = true;
  errno = 0;
  while (const struct dirent* entry = ::readdir(stream)) {
    if (std::strcmp(entry->d_name, ".") != 0 &&
        std::strcmp(entry->d_name, "..") != 0) {
      empty = false;
      break;
    }
  }
  const int read_error = errno;
  if (::closedir(stream) != 0) {
    error = "cannot close reserved output directory: " +
            std::string(std::strerror(errno));
    return false;
  }
  if (read_error != 0) {
    error = "cannot inspect reserved output directory: " +
            std::string(std::strerror(read_error));
    return false;
  }
  if (!empty) {
    error = "reserved output directory is not empty";
    return false;
  }
  return true;
}

bool file_matches_identity(int directory_fd,
                           const char* name,
                           const FileIdentity& identity) {
  if (!identity.valid) return false;
  struct stat info;
  return ::fstatat(directory_fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0 &&
         info.st_dev == identity.device && info.st_ino == identity.inode;
}

void remove_owned_file(int directory_fd,
                       const char* name,
                       const FileIdentity& identity) {
  if (file_matches_identity(directory_fd, name, identity)) {
    ::unlinkat(directory_fd, name, 0);
  }
}

void cleanup_failed_output(int directory_fd,
                           const FileIdentity& csv_temporary,
                           const FileIdentity& status_temporary,
                           const FileIdentity& csv_published,
                           const FileIdentity& status_published) {
  remove_owned_file(directory_fd, "collection.status.json", status_published);
  remove_owned_file(directory_fd, "collection.csv", csv_published);
  remove_owned_file(directory_fd, ".collection.status.json.tmp",
                    status_temporary);
  remove_owned_file(directory_fd, ".collection.csv.tmp", csv_temporary);
}

bool publish_temporary_file(int directory_fd,
                            const char* temporary_name,
                            const char* published_name,
                            const FileIdentity& temporary_identity,
                            FileIdentity& published_identity,
                            const char* label,
                            std::string& error) {
  if (!file_matches_identity(directory_fd, temporary_name,
                             temporary_identity)) {
    error = std::string(label) + " temporary file changed before publication";
    return false;
  }
  if (::linkat(directory_fd, temporary_name, directory_fd, published_name,
               0) != 0) {
    error = "cannot publish collection " + std::string(label) + ": " +
            std::string(std::strerror(errno));
    return false;
  }
  published_identity = temporary_identity;
  remove_owned_file(directory_fd, temporary_name, temporary_identity);
  return true;
}

int open_output_directory(const std::string& directory, std::string& error) {
  const int fd = ::open(directory.c_str(),
                        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    error = "cannot open output directory without following links: " +
            std::string(std::strerror(errno));
    return -1;
  }
  struct stat info;
  if (::fstat(fd, &info) != 0 || !S_ISDIR(info.st_mode)) {
    const int inspect_error = errno;
    ::close(fd);
    error = "output is not a directory: " +
            std::string(std::strerror(inspect_error));
    return -1;
  }
  return fd;
}

}  // namespace

CollectionWriteResult write_pace_collection(
    const CollectionWriteRequest& request,
    const PaceSampleBuffer& buffer,
    const PaceCollector& collector) {
  if (request.output_directory.empty()) {
    return failed_result("output directory must not be empty");
  }
  if (!std::isfinite(request.dt_s) || request.dt_s != kExpectedDtS) {
    return failed_result("dt_s must equal the frozen 0.002 second period");
  }
  if (request.collection_mode != "simulation" &&
      request.collection_mode != "execute") {
    return failed_result("collection_mode must be simulation or execute");
  }
  if (request.low_state_tick_step == 0) {
    return failed_result("low_state_tick_step must be positive");
  }
  bool reserved_output_directory = false;
  if (::mkdir(request.output_directory.c_str(), 0755) != 0) {
    const int mkdir_error = errno;
    if (mkdir_error != EEXIST || request.collection_mode != "execute") {
      if (mkdir_error == EEXIST) {
        return failed_result("output directory already exists: " +
                             request.output_directory);
      }
      return failed_result("cannot create output directory: " +
                           std::string(std::strerror(mkdir_error)));
    }
    reserved_output_directory = true;
  }

  const std::string csv_path =
      join_path(request.output_directory, "collection.csv");
  const std::string status_path =
      join_path(request.output_directory, "collection.status.json");
  std::string error;
  const int output_fd = open_output_directory(request.output_directory, error);
  if (output_fd < 0) return failed_result(error);
  if (reserved_output_directory && !directory_is_empty(output_fd, error)) {
    ::close(output_fd);
    return failed_result(error);
  }

  std::string status_text;
  FileIdentity csv_temporary;
  FileIdentity status_temporary;
  FileIdentity csv_published;
  FileIdentity status_published;
  if (!make_status_json(request, buffer, collector, status_text, error) ||
      !write_csv_file(output_fd, ".collection.csv.tmp", buffer,
                      csv_temporary, error) ||
      !write_text_file(output_fd, ".collection.status.json.tmp", status_text,
                       status_temporary, error)) {
    if (!reserved_output_directory) {
      cleanup_failed_output(output_fd, csv_temporary, status_temporary,
                            csv_published, status_published);
    }
    ::close(output_fd);
    return failed_result(error);
  }

  if (!publish_temporary_file(output_fd, ".collection.csv.tmp",
                              "collection.csv", csv_temporary,
                              csv_published, "CSV", error)) {
    if (!reserved_output_directory) {
      cleanup_failed_output(output_fd, csv_temporary, status_temporary,
                            csv_published, status_published);
    }
    ::close(output_fd);
    return failed_result(error);
  }
  if (!publish_temporary_file(output_fd, ".collection.status.json.tmp",
                              "collection.status.json", status_temporary,
                              status_published, "status", error)) {
    // A pre-reserved physical directory may contain irreplaceable raw data.
    // Preserve every owned artifact for recovery on any publication failure.
    if (!reserved_output_directory) {
      cleanup_failed_output(output_fd, csv_temporary, status_temporary,
                            csv_published, status_published);
    }
    ::close(output_fd);
    return failed_result(error);
  }
  ::close(output_fd);

  CollectionWriteResult result;
  result.success = true;
  result.csv_path = csv_path;
  result.status_path = status_path;
  result.error.clear();
  return result;
}

}  // namespace a1_base
