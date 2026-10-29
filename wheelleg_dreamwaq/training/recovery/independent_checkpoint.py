from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import torch
import numpy as np

from .checkpoint import capture_global_rng_state, restore_global_rng_state
from .contracts import file_sha256, stable_hash


IDENTITY_HASH_KEYS = ("source_sha256", "config_sha256", "contract_sha256", "reference_sha256",
                      "calibration_sha256", "geometry_sha256")


def validate_identity(identity):
    if identity.get("schema_version") != "RecoveryIndependentStandV1":
        raise ValueError("checkpoint 仅接受独立站立身份")
    for key in IDENTITY_HASH_KEYS:
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(letter not in "0123456789abcdefABCDEF" for letter in value):
            raise ValueError(f"checkpoint 身份哈希无效: {key}")


def validate_finite(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("checkpoint 张量非有限")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all():
            raise ValueError("checkpoint numpy 数组非有限")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint 标量非有限")
    elif isinstance(value, dict):
        for item in value.values():
            validate_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_finite(item)


def save_independent_checkpoint(path, payload, identity):
    validate_identity(identity)
    validate_finite(payload)
    barrier = payload["logging_barrier"]
    if (payload["completed_updates"] < 0 or barrier["pending"] != 0
            or barrier["submitted"] != barrier["consumed"]
            or barrier["last_sequence"] != barrier["consumed"]
            or barrier["last_update"] != payload["completed_updates"]
            or barrier["finalized"] or barrier["worker_state"] != "synchronous"):
        raise ValueError("checkpoint 更新或日志确认状态无效")
    destination = Path(path)
    index_path = Path(str(destination) + ".index.json")
    if destination.exists() or index_path.exists():
        raise FileExistsError("checkpoint 路径必须全新")
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": "IndependentStandingCheckpointV1", "identity": deepcopy(identity),
                "payload": payload, "global_rng_state": capture_global_rng_state()}
    with destination.open("xb") as stream:
        torch.save(envelope, stream)
        stream.flush()
        os.fsync(stream.fileno())
    index = {"schema_version": "IndependentStandingCheckpointIndexV1", "checkpoint_file": destination.name,
             "checkpoint_sha256": file_sha256(destination), "identity": deepcopy(identity),
             "identity_sha256": stable_hash(identity), "completed_updates": payload["completed_updates"]}
    with index_path.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return index


def load_independent_checkpoint(path, identity, *, restore_rng=True, include_rng=False):
    validate_identity(identity)
    source = Path(path)
    index = json.loads(Path(str(source) + ".index.json").read_text(encoding="utf-8"))
    expected = {"schema_version", "checkpoint_file", "checkpoint_sha256", "identity", "identity_sha256", "completed_updates"}
    if set(index) != expected or index["schema_version"] != "IndependentStandingCheckpointIndexV1":
        raise ValueError("checkpoint index 版本或字段不完整")
    if index["checkpoint_file"] != source.name or index["identity"] != identity or index["identity_sha256"] != stable_hash(identity):
        raise ValueError("checkpoint index 身份不一致")
    if file_sha256(source) != index["checkpoint_sha256"]:
        raise ValueError("checkpoint 内容哈希不一致")
    envelope = torch.load(source, map_location="cpu", weights_only=False)
    if set(envelope) != {"schema_version", "identity", "payload", "global_rng_state"} or envelope["schema_version"] != "IndependentStandingCheckpointV1":
        raise ValueError("checkpoint 模型版本或字段不完整")
    if envelope["identity"] != identity or envelope["payload"]["completed_updates"] != index["completed_updates"]:
        raise ValueError("checkpoint 内部身份不一致")
    validate_finite(envelope["payload"])
    if restore_rng:
        restore_global_rng_state(envelope["global_rng_state"])
    return (envelope["payload"], envelope["global_rng_state"]) if include_rng else envelope["payload"]
