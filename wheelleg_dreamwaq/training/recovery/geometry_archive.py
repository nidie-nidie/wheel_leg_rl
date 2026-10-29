import hashlib
from pathlib import Path


def prefix_sha256(path, byte_cutoff):
    if not isinstance(byte_cutoff, int) or byte_cutoff < 0:
        raise ValueError("几何归档截面必须是非负字节数")
    digest = hashlib.sha256()
    remaining = byte_cutoff
    with Path(path).open("rb") as stream:
        while remaining:
            payload = stream.read(min(remaining, 1024 * 1024))
            if not payload:
                raise ValueError("几何归档长度小于已保存截面")
            digest.update(payload)
            remaining -= len(payload)
    return digest.hexdigest().upper()


def verify_geometry_barrier(barrier):
    if (barrier["schema_version"] != "RecoveryGeometryBarrierV1"
            or barrier["all_requests_acknowledged"] is not True
            or barrier["pending_requests"] != 0
            or barrier["archived_diagnostic_sequence"] > barrier["submitted_sequence"]):
        raise ValueError("几何保存屏障状态无效")
    if prefix_sha256(barrier["archive_path"], barrier["byte_cutoff"]) != barrier["prefix_sha256"]:
        raise ValueError("几何归档已保存截面哈希不匹配")
    return dict(barrier)
