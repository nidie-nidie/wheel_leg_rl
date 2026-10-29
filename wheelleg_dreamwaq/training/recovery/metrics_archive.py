from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from .contracts import file_sha256, stable_hash


class MetricsArchive:
    def __init__(self, directory: str | Path, *, segment_records: int = 256) -> None:
        if not isinstance(segment_records, int) or segment_records <= 0:
            raise ValueError("Archive segment record count must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.archive_id = uuid4().hex
        self.segment_records = segment_records
        self._segments: list[dict[str, Any]] = []
        self._sequence = 0
        self._segment_number = 0
        self._snapshot_number = 0
        self._stream = None
        self._raw = None
        self._current_path: Path | None = None
        self._current_count = 0
        self._current_start = 0
        self._last_index: dict | None = None
        self._accepted_archive_ids = {self.archive_id}
        self._sidecars: list[dict] = []

    @classmethod
    def from_index(cls, descriptor: dict, directory: str | Path) -> MetricsArchive:
        index = cls.verify_index(descriptor)
        archive = cls(directory, segment_records=index["segment_records"])
        archive._segments = list(index["segments"])
        archive._sidecars = list(index.get("sidecars", []))
        archive._sequence = index["last_sequence"]
        archive._accepted_archive_ids.update(index["accepted_archive_ids"])
        archive.append("resume_origin", "all", {"source_index": descriptor})
        return archive

    @staticmethod
    def verify_index(descriptor: dict) -> dict:
        if set(descriptor) != {"schema_version", "index_path", "index_sha256", "archive_id", "last_sequence"}:
            raise ValueError("Metrics archive descriptor fields are invalid")
        if descriptor["schema_version"] != "RecoveryMetricsArchiveIndexRefV1":
            raise ValueError("Unsupported metrics archive reference")
        path = Path(descriptor["index_path"])
        if file_sha256(path) != descriptor["index_sha256"]:
            raise ValueError("Metrics archive index SHA256 changed")
        index = json.loads(path.read_text(encoding="utf-8"))
        if index["schema_version"] != "RecoveryMetricsArchiveIndexV1":
            raise ValueError("Unsupported metrics archive index")
        if index["archive_id"] != descriptor["archive_id"] or index["last_sequence"] != descriptor["last_sequence"]:
            raise ValueError("Metrics archive index identity changed")
        expected_start = 1
        for segment in index["segments"]:
            source = Path(segment["path"])
            if source.stat().st_size != segment["size_bytes"] or file_sha256(source) != segment["sha256"]:
                raise ValueError("Metrics archive segment changed or is incomplete")
            if segment["first_sequence"] != expected_start:
                raise ValueError("Metrics archive segment sequence has a gap")
            if segment["last_sequence"] - segment["first_sequence"] + 1 != segment["records"]:
                raise ValueError("Metrics archive segment count is inconsistent")
            expected_start = segment["last_sequence"] + 1
        if expected_start - 1 != index["last_sequence"]:
            raise ValueError("Metrics archive index final sequence is inconsistent")
        for sidecar in index.get("sidecars", []):
            source = Path(sidecar["path"])
            if source.stat().st_size != sidecar["size_bytes"] or file_sha256(source) != sidecar["sha256"]:
                raise ValueError("Metrics archive binary source packet changed")
        return index

    def append_tensor_packets(self, columns: np.ndarray, headers: list[dict], protocol: dict) -> dict:
        if columns.dtype != np.int64 or columns.ndim != 3 or len(columns) != len(headers):
            raise ValueError("原始日志packet必须为固定int64三维数组")
        directory = self.directory / "tensor_packets"
        directory.mkdir(exist_ok=True)
        path = directory / f"packets_{len(self._sidecars) + 1:08d}.npy.gz"
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0) as stream:
                np.save(stream, columns, allow_pickle=False)
            raw.flush()
            os.fsync(raw.fileno())
        descriptor = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_sha256(path),
                      "shape": list(columns.shape), "dtype": "int64", "headers": headers,
                      "protocol_sha256": protocol["sha256"]}
        self._sidecars.append(descriptor)
        self.append("tensor_packets", "all", descriptor)
        return descriptor

    def accepts_archive_id(self, archive_id: str) -> bool:
        return archive_id in self._accepted_archive_ids

    @property
    def last_sequence(self) -> int:
        return self._sequence

    def append(self, kind: str, stream_id: str, payload: dict) -> int:
        if not kind or not isinstance(stream_id, str) or not stream_id:
            raise ValueError("Metrics archive kind and stream ID are required")
        if self._stream is None:
            self._segment_number += 1
            self._current_path = self.directory / f"segment_{self._segment_number:07d}.jsonl.gz"
            self._raw = self._current_path.open("xb")
            self._stream = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, compresslevel=1, mtime=0)
            self._current_start = self._sequence + 1
            self._current_count = 0
        sequence = self._sequence + 1
        record = {"schema_version": "RecoveryMetricsRecordV1", "sequence": sequence,
                  "archive_id": self.archive_id, "kind": kind, "stream_id": stream_id, "payload": payload}
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self._stream.write(encoded)
        self._sequence = sequence
        self._current_count += 1
        self._last_index = None
        if self._current_count >= self.segment_records:
            self._seal_segment()
        return sequence

    def _seal_segment(self) -> None:
        if self._stream is None:
            return
        self._stream.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._segments.append({
            "path": str(self._current_path), "size_bytes": self._current_path.stat().st_size,
            "sha256": file_sha256(self._current_path), "records": self._current_count,
            "first_sequence": self._current_start, "last_sequence": self._sequence,
        })
        self._stream = None
        self._raw = None
        self._current_path = None

    def snapshot(self) -> dict:
        if self._last_index is not None:
            return dict(self._last_index)
        self._seal_segment()
        self._snapshot_number += 1
        path = self.directory / f"index_{self._snapshot_number:07d}.json"
        payload = {"schema_version": "RecoveryMetricsArchiveIndexV1", "archive_id": self.archive_id,
                   "accepted_archive_ids": sorted(self._accepted_archive_ids),
                   "segment_records": self.segment_records, "last_sequence": self._sequence,
                   "segments": list(self._segments), "sidecars": list(self._sidecars)}
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._last_index = {"schema_version": "RecoveryMetricsArchiveIndexRefV1", "index_path": str(path),
                            "index_sha256": file_sha256(path), "archive_id": self.archive_id,
                            "last_sequence": self._sequence}
        return dict(self._last_index)

    def iter_records(self, *, stream_id: str | None = None) -> Iterator[dict]:
        self._seal_segment()
        for segment in self._segments:
            with gzip.open(segment["path"], "rt", encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    if stream_id is None or record["stream_id"] == stream_id:
                        yield record

    def close(self) -> None:
        self._seal_segment()
