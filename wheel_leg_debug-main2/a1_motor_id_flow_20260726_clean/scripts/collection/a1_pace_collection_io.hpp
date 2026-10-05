#pragma once

#include "a1_base/pace_collector.hpp"

#include <cstdint>
#include <string>

namespace a1_base {

struct CollectionWriteRequest {
  std::string output_directory;
  std::string collection_mode;
  double dt_s;
  std::uint32_t low_state_tick_step;
  bool trial_valid;
  PaceStopReason stop_reason;
};

struct CollectionWriteResult {
  bool success;
  std::string csv_path;
  std::string status_path;
  std::string error;
};

CollectionWriteResult write_pace_collection(
    const CollectionWriteRequest& request,
    const PaceSampleBuffer& buffer,
    const PaceCollector& collector);

}  // namespace a1_base
