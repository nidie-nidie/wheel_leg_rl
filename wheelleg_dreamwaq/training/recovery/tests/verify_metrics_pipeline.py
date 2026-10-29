from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import torch

from recovery.batched_supervisor import BatchedRecoverySupervisor, BLOCKER_BITS
from recovery.checkpoint import read_checkpoint_envelope, validate_logging_barrier
from recovery.contracts import ARTIFACT_ROOT, file_sha256, source_sha256
from recovery.metrics import AttemptContext
from recovery.metrics_stream import RecoveryMetricsStream


def component(directory: Path, device: str, count: int) -> dict:
    torch.set_num_threads(1)
    bank_path = ARTIFACT_ROOT / "bootstrap_pose_bank_v1/pose_bank.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    records = [record for record in bank["records"] if record["split"] == "train"]
    contexts, ids, values = {}, [], []
    for index in range(count):
        record = records[index % len(records)]
        key = (record["initial_category"], record["motion_kind"])
        context_id = next((item for item, context in contexts.items()
                           if (context.initial_category, context.motion_kind) == key), None)
        if context_id is None:
            context_id = len(contexts)
            contexts[context_id] = AttemptContext(*key, "RecoveryBootstrapNominalV1", "independent")
        ids.append(context_id)
        frame = record["clean_history"][-1]
        tilt = torch.tensor(-frame[5], dtype=torch.float64).clamp(-1, 1).acos().item()
        values.append([tilt, *frame[:3], *frame[6:10], *frame[10:16], record["metadata"]["com_height_m"]])
    stream = RecoveryMetricsStream(count, device, directory / "stream", contexts)
    supervisor = BatchedRecoverySupervisor(count, device, standalone=True,
        context_ids=torch.tensor(ids, device=device), defer_initial_begin=True)
    frames = torch.zeros(count, dtype=torch.int64, device=device)
    stream.submit(supervisor.reset(frame_ids=frames), kind="reset")
    observations = torch.tensor(values, dtype=torch.float32, device=device)
    kwargs = {"wheel_support": torch.zeros(count, 2, dtype=torch.bool, device=device),
              "body_support": torch.ones(count, dtype=torch.bool, device=device),
              "action_compatible": torch.zeros(count, dtype=torch.bool, device=device),
              "compatibility_mask": torch.full((count,), BLOCKER_BITS["calibration_unavailable"], dtype=torch.int64, device=device),
              "physical_failure": torch.zeros(count, dtype=torch.int64, device=device)}
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for _ in range(24):
        frames += 1
        stream.submit(supervisor.post_step(observations, frames, **kwargs))
    submitted = perf_counter()
    stream.submit_rollout()
    stream.flush()
    drained = perf_counter()
    checkpoint = stream.checkpoint(frames)
    saved = perf_counter()
    validate_logging_barrier({**checkpoint, "frame_ids": frames.cpu()})
    final = stream.close(frames)
    ended = perf_counter()
    result = {"source": "48条真实训练快照重复构造24帧组件负载；无连续物理轨迹、仿真或PPO",
        "device": device, "num_envs": count, "post_step_calls": count * 24, "source_bank_file_sha256": file_sha256(bank_path),
        "submit_fsm_and_transport_seconds": submitted - started, "including_drain_seconds": drained - started,
        "checkpoint_censor_snapshot_seconds": saved - drained, "final_administrative_censor_and_close_seconds": ended - saved,
        "worker_processing_seconds": checkpoint["logging_barrier"]["worker_processing_seconds"],
        "slot_budget": stream.budget, "stream_statistics": stream.statistics,
        "checkpoint_barrier": checkpoint["logging_barrier"], "final_archive": final["metrics_archive"],
        "checkpoint_denominator": sum(row["legal_initial_count"] for state in checkpoint["metrics"] for row in state["aggregates"]),
        "final_denominator": sum(row["legal_initial_count"] for state in final["metrics"] for row in state["aggregates"])}
    if torch.device(device).type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["host_memory_stats_available"] = hasattr(torch.cuda.memory, "host_memory_stats")
        if result["host_memory_stats_available"]:
            result["host_memory_stats"] = torch.cuda.memory.host_memory_stats()
    return result


def legacy(directory: Path, path: Path) -> dict:
    manifest = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    started = perf_counter()
    envelope = read_checkpoint_envelope(path, manifest["identity"])
    state = envelope["payload"]["environment_state"]
    source_contexts = [AttemptContext(**row["context"]) for metric in state["metrics"] for row in metric["aggregates"]]
    unique = list(dict.fromkeys(source_contexts))
    print("METRICS_LEGACY_RAW_RECOMPUTATION_START", flush=True)
    stream = RecoveryMetricsStream(len(state["metrics"]), "cpu", directory / "legacy", dict(enumerate(unique)), restore_state=state)
    verified = perf_counter()
    ids = stream.next_ids()
    if ids["next_event_id"] != [item["next_event_id"] for item in state["metrics"]]:
        raise ValueError("旧checkpoint导入event IDs改变")
    if ids["next_attempt_id"] != [item["next_attempt_id"] for item in state["metrics"]]:
        raise ValueError("旧checkpoint导入attempt IDs改变")
    snapshot = stream.close(state["frame_ids"])
    if [item["aggregates"] for item in snapshot["metrics"]] != [item["aggregates"] for item in state["metrics"]]:
        raise ValueError("旧checkpoint导入重复计算分母或混入截面后记录")
    return {"parent_checkpoint": str(path.resolve()), "parent_checkpoint_sha256": file_sha256(path),
        "completed_updates": envelope["payload"]["completed_updates"], "num_envs": len(state["metrics"]),
        "source_archive": state["metrics_archive"], "all_stream_raw_recomputation_passed": True,
        "next_ids_preserved": True, "aggregate_and_censored_denominators_preserved": True,
        "cpu_load_plus_recompute_seconds": verified - started, "total_seconds": perf_counter() - started,
        "child_process_exited": stream._process.poll() == 0, "cuda_initialized": torch.cuda.is_initialized()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=6144)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    directory = ARTIFACT_ROOT / "performance-validation-20261023" / f"logging_{datetime.now().strftime('%H%M%S_%f')}"
    directory.mkdir(parents=True)
    result = {"schema_version": "RecoveryMetricsPipelineVerificationV1", "source_sha256": source_sha256()}
    if not args.legacy_only:
        result["component"] = component(directory, args.device, args.num_envs)
    if args.legacy_checkpoint:
        result["legacy"] = legacy(directory, args.legacy_checkpoint)
    result["passed"] = True
    path = directory / "verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("METRICS_VERIFICATION " + str(path), flush=True)
    print(json.dumps({key: value for key, value in result.items() if key != "component"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
