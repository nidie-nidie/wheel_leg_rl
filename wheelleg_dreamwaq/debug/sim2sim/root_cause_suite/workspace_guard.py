from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.exists():
        return candidate.resolve(strict=True)
    missing_parts: list[str] = []
    ancestor = candidate
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise FileNotFoundError(f"No existing ancestor for {candidate}")
        missing_parts.append(ancestor.name)
        ancestor = ancestor.parent
    resolved = ancestor.resolve(strict=True)
    for part in reversed(missing_parts):
        resolved = resolved / part
    return resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class WorkspaceGuard:
    def __init__(self, writable_root: str | Path) -> None:
        self.writable_root = Path(writable_root).resolve(strict=True)

    def require_writable(self, path: str | Path) -> Path:
        raw = Path(path)
        if ".." in raw.parts:
            raise PermissionError(f"Parent traversal is forbidden: {raw}")
        resolved = canonical_path(raw)
        if not _inside(resolved, self.writable_root):
            raise PermissionError(f"Write path escapes suite root: {resolved}")
        return resolved

    def atomic_write_json(self, path: str | Path, payload: Any) -> Path:
        destination = self.require_writable(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


def snapshot_tree(root: str | Path, *, exclude: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    base = Path(root).resolve()
    excluded = set(exclude)
    snapshot: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return snapshot
    files = sorted((item for item in base.rglob("*") if item.is_file()), key=lambda p: p.as_posix())
    for path in files:
        relative = path.relative_to(base).as_posix()
        if any(relative == item or relative.startswith(item.rstrip("/") + "/") for item in excluded):
            continue
        stat_result = path.stat()
        snapshot[relative] = {
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return snapshot


def compare_snapshots(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            changes.append({"path": path, "before": before.get(path), "after": after.get(path)})
    return changes
