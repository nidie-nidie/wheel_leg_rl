from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping


DESIGN_VERSION = "RootCauseSuiteCoreV1.17"
DESIGN_SHA256 = "8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD"
SCHEMA_VERSION = "RootCauseSuiteRunV1"
SUITE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SUITE_ROOT.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DESIGN_PATH = SUITE_ROOT / "2026-10-07-sim2sim-root-cause-suite-design.md"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DECLARED_INFINITY_SCHEMA_VERSION = "RootCauseDeclaredInfinityV1"
DECLARED_INFINITY_SOURCE = "composed_usd_attribute"
DECLARED_INFINITY_TAG_KEYS = frozenset(
    {"schema_version", "kind", "semantics", "source", "attribute"}
)
ISAAC_LOOP_CLOSURE_PRIM_PATHS = (
    "/World/envs/env_0/Robot/jIO/jIO_loop_closure",
    "/World/envs/env_0/Robot/jKN/jKN_loop_closure",
    "/World/envs/env_0/Robot/jEC/jAG_loop_closure",
    "/World/envs/env_0/Robot/jCF/jCF_revolute_joint",
)


class BootstrapError(RuntimeError):
    """Raised when the suite is launched outside the frozen PowerShell entry."""


class ConfigurationError(ValueError):
    """CLI or caller supplied an invalid suite configuration."""


class IdentityDriftError(ValueError):
    """A frozen formal or runtime identity no longer matches the contract."""


class StageExecutionError(RuntimeError):
    """A worker failed or a required stage did not complete."""


class EvidenceIntegrityError(ValueError):
    """Evidence schema, hashes, or causal identity are internally inconsistent."""


@dataclass(frozen=True)
class DeclaredInfinitySpec:
    kind: str
    semantics: str
    attribute: str
    expected_shape: tuple[int, ...]


def _usd_spec(
    attribute: str,
    kind: str,
    semantics: str,
    expected_shape: tuple[int, ...] = (),
) -> DeclaredInfinitySpec:
    return DeclaredInfinitySpec(
        kind=kind,
        semantics=semantics,
        attribute=attribute,
        expected_shape=expected_shape,
    )


_declared_usd_infinity_specs: dict[tuple[str, str], DeclaredInfinitySpec] = {
    ("/physicsScene", "physxScene:maxBiasCoefficient"): _usd_spec(
        "physxScene:maxBiasCoefficient",
        "positive_infinity",
        "unbounded_solver_limit",
    ),
    ("/World/envs/env_0/Coupon", "physics:centerOfMass"): _usd_spec(
        "physics:centerOfMass",
        "negative_infinity",
        "usd_schema_default_unset",
        (3,),
    ),
    ("/World/envs/env_0/Coupon", "physxRigidBody:maxContactImpulse"): _usd_spec(
        "physxRigidBody:maxContactImpulse",
        "positive_infinity",
        "unbounded_contact_impulse",
    ),
    ("/World/envs/env_0/Coupon", "physxRigidBody:maxLinearVelocity"): _usd_spec(
        "physxRigidBody:maxLinearVelocity",
        "positive_infinity",
        "unbounded_linear_velocity",
    ),
}
for _closure_path in ISAAC_LOOP_CLOSURE_PRIM_PATHS:
    _declared_usd_infinity_specs[(_closure_path, "physics:lowerLimit")] = _usd_spec(
        "physics:lowerLimit",
        "negative_infinity",
        "unbounded_joint_lower_limit",
    )
    _declared_usd_infinity_specs[(_closure_path, "physics:upperLimit")] = _usd_spec(
        "physics:upperLimit",
        "positive_infinity",
        "unbounded_joint_upper_limit",
    )
DECLARED_USD_INFINITY_SPECS: Mapping[
    tuple[str, str], DeclaredInfinitySpec
] = MappingProxyType(_declared_usd_infinity_specs)
del _closure_path, _declared_usd_infinity_specs


def declared_usd_infinity_spec(
    prim_path: str, attribute: str
) -> DeclaredInfinitySpec | None:
    return DECLARED_USD_INFINITY_SPECS.get((str(prim_path), str(attribute)))


def declared_infinity_tag(spec: DeclaredInfinitySpec) -> dict[str, str]:
    return {
        "schema_version": DECLARED_INFINITY_SCHEMA_VERSION,
        "kind": spec.kind,
        "semantics": spec.semantics,
        "source": DECLARED_INFINITY_SOURCE,
        "attribute": spec.attribute,
    }


def _plain_data(value: Any) -> Any:
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return _plain_data(value.tolist())
        if isinstance(value, np.generic):
            return _plain_data(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    return value


def _nested_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    child_shapes = {_nested_shape(item) for item in value}
    if len(child_shapes) > 1:
        raise EvidenceIntegrityError("Declared USD infinity value is ragged")
    child_shape = next(iter(child_shapes), ())
    return (len(value), *child_shape)


def _real_leaves(value: Any) -> list[float]:
    if isinstance(value, list):
        leaves: list[float] = []
        for item in value:
            leaves.extend(_real_leaves(item))
        return leaves
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceIntegrityError(
            "Declared USD infinity value contains a non-numeric leaf"
        )
    return [float(value)]


def encode_declared_usd_value(
    value: Any, *, prim_path: str, attribute: str
) -> Any:
    """Encode only frozen composed-USD infinity sentinels."""
    normalized = _plain_data(value)

    def nonfinite(item: Any) -> list[float]:
        if isinstance(item, Mapping):
            values: list[float] = []
            for child in item.values():
                values.extend(nonfinite(child))
            return values
        if isinstance(item, list):
            values = []
            for child in item:
                values.extend(nonfinite(child))
            return values
        if isinstance(item, float) and not math.isfinite(item):
            return [item]
        return []

    invalid = nonfinite(normalized)
    if any(math.isnan(item) for item in invalid):
        raise EvidenceIntegrityError(
            f"Composed USD attribute {prim_path}:{attribute} contains NaN"
        )
    if not invalid:
        return normalized
    spec = declared_usd_infinity_spec(prim_path, attribute)
    if spec is None:
        raise EvidenceIntegrityError(
            f"Undeclared USD infinity at {prim_path}:{attribute}"
        )
    actual_shape = _nested_shape(normalized)
    if actual_shape != spec.expected_shape:
        raise EvidenceIntegrityError(
            f"Declared USD infinity shape drifted at {prim_path}:{attribute}: "
            f"actual={actual_shape}, expected={spec.expected_shape}"
        )
    leaves = _real_leaves(normalized)
    expected_positive = spec.kind == "positive_infinity"
    if not leaves or any(
        not math.isinf(item) or (item > 0.0) is not expected_positive
        for item in leaves
    ):
        raise EvidenceIntegrityError(
            f"Declared USD infinity sentinel drifted at {prim_path}:{attribute}"
        )
    tag = declared_infinity_tag(spec)

    def replace(item: Any) -> Any:
        if isinstance(item, list):
            return [replace(child) for child in item]
        return dict(tag)

    return replace(normalized)


def validate_declared_infinity_payload(
    payload: Any,
    *,
    declaration_for_path: Callable[
        [tuple[str | int, ...]], DeclaredInfinitySpec | None
    ],
    label: str,
) -> None:
    """Reject raw non-finite values and tags outside a caller-owned exact path contract."""
    shaped_tags: dict[
        tuple[tuple[str | int, ...], DeclaredInfinitySpec],
        set[tuple[int, ...]],
    ] = {}

    def visit(value: Any, path: tuple[str | int, ...]) -> None:
        np = sys.modules.get("numpy")
        if np is not None:
            if isinstance(value, np.ndarray):
                visit(value.tolist(), path)
                return
            if isinstance(value, np.generic):
                visit(value.item(), path)
                return
        if isinstance(value, Mapping):
            keys = {str(key) for key in value}
            spec = declaration_for_path(path)
            if spec is not None:
                if keys != DECLARED_INFINITY_TAG_KEYS:
                    raise EvidenceIntegrityError(
                        f"{label} contains a malformed infinity tag at {path}: "
                        f"keys={sorted(keys)}"
                    )
                if dict(value) != declared_infinity_tag(spec):
                    raise EvidenceIntegrityError(
                        f"{label} contains an invalid infinity tag at {path}"
                    )
                if spec.expected_shape:
                    rank = len(spec.expected_shape)
                    suffix = path[-rank:]
                    if len(path) < rank or not all(
                        isinstance(index, int) for index in suffix
                    ):
                        raise EvidenceIntegrityError(
                            f"{label} contains a shaped infinity tag at an invalid path: {path}"
                        )
                    key = (path[:-rank], spec)
                    shaped_tags.setdefault(key, set()).add(
                        tuple(int(index) for index in suffix)
                    )
                return
            tag_like = (
                {"kind", "semantics"}.issubset(keys)
                or {"kind", "attribute"}.issubset(keys)
                or len(keys & DECLARED_INFINITY_TAG_KEYS) >= 3
                or str(value.get("schema_version", "")).startswith(
                    "RootCauseDeclaredInfinity"
                )
            )
            if tag_like:
                if keys != DECLARED_INFINITY_TAG_KEYS:
                    raise EvidenceIntegrityError(
                        f"{label} contains a malformed infinity tag at {path}: "
                        f"keys={sorted(keys)}"
                    )
                raise EvidenceIntegrityError(
                    f"{label} contains an infinity tag at an undeclared path: {path}"
                )
            for key, child in value.items():
                visit(child, (*path, str(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, index))
            return
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceIntegrityError(
                f"{label} contains raw NaN/Inf at {path}"
            )

    visit(payload, ())
    for (base_path, spec), actual in shaped_tags.items():
        expected = set(product(*(range(size) for size in spec.expected_shape)))
        if actual != expected:
            raise EvidenceIntegrityError(
                f"{label} contains an incomplete shaped infinity sentinel at {base_path}: "
                f"actual={sorted(actual)}, expected={sorted(expected)}"
            )


class StageName(str, Enum):
    G00_INTEGRITY = "G00_integrity"
    G01_REPEATABILITY = "G01_repeatability"
    G02_ADAPTER = "G02_adapter"
    G03_INSTRUMENTATION = "G03_instrumentation"
    P10_REST = "P10_rest"
    P20_STATIC_PROPERTIES = "P20_static_properties"
    P30_ACTUATOR = "P30_actuator"
    P40_CLOSURE = "P40_closure"
    P50_CONTACT = "P50_contact"
    P60_FULL_ROBOT = "P60_full_robot"
    C70_CHECKPOINT = "C70_checkpoint"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


class CandidateStatus(str, Enum):
    SUPPORTED_PRIMARY = "supported_primary"
    SUPPORTED_CONTRIBUTOR = "supported_contributor"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class FrozenFile:
    relative_path: str
    sha256: str

    def absolute_path(self) -> Path:
        return PROJECT_ROOT / self.relative_path


FROZEN_POLICIES: Mapping[str, Mapping[str, FrozenFile]] = {
    "dreamwaq_run01": {
        "actor": FrozenFile(
            "artifacts/phase2_dreamwaq/training-suite-20261007-185711/"
            "run-01-evaluation/export/actor.ts",
            "4C2C64C2D9F9F88532FCEFF4AE45AA049F970646761D2D0ACE0A29678621D6C7",
        ),
        "manifest": FrozenFile(
            "artifacts/phase2_dreamwaq/training-suite-20261007-185711/"
            "run-01-evaluation/export/policy_manifest.json",
            "322232B78674217579E491FF9B88E1E688DB899A2C5064830F0D3149B0E992C3",
        ),
    },
    "phase1r_run03": {
        "actor": FrozenFile(
            "artifacts/phase1_randomized_v3/training-suite-20261007-062147/"
            "exports/run-03/actor.ts",
            "BFA655DF083EC8634DE18E4966E85A64219752F4E2875D3B2548BC7AA0E70AC6",
        ),
        "manifest": FrozenFile(
            "artifacts/phase1_randomized_v3/training-suite-20261007-062147/"
            "exports/run-03/policy_manifest.json",
            "53866E1078CFA2E2E337578ECC7DEE1A0E804B2D0D2892C90BA8FD7D6D9ADE40",
        ),
    },
}


FROZEN_REPLAY_SOURCE: Mapping[str, Any] = {
    "source_scenario_id": "P60_D_REPLAY_SOURCE_DREAMWAQ_RUN01",
    "reset_cache": FrozenFile(
        "artifacts/phase2_dreamwaq/training-suite-20261007-185711/"
        "run-01-evaluation/isaac/evaluation-reset-cache.pt",
        "008EE5D50E1C622D4AC0408E3C5E1C0B8CFCA618ABD3163E221A39D203B3DECE",
    ),
    "evaluation_summary": FrozenFile(
        "artifacts/phase2_dreamwaq/training-suite-20261007-185711/"
        "run-01-evaluation/isaac/dreamwaq-run01/summary.json",
        "3F8FA420604B65A514CB767CF770DA6F00E48610C8832B5487AA3A02AFCDB9BA",
    ),
    "evaluation_contract_hash": "8A3D9CC722B7F947B6C9D0DA8E4B68B90877D40E990FAE8D00495D5E5A7CAD80",
    "reset_cache_tensor_sha256": "446817DB7509AAC24F66D7111391FF9223D50FA8B33469C31CFF9B79CE3DF21D",
    "reset_cache_identity_hash": "D87209736F3B070EB3DB3B93879457DB7A26A5098E132F6CD00181826E05A8A4",
    "reset_cache_schema": "ClosedChainResetCacheSchemaV2",
    "reset_cache_relaxation_algorithm": "BoundaryClampedPhysXRelaxationV1",
    "reset_cache_root_height_algorithm": "ClosedChainRootHeightAlignmentV1",
    "environment_count": 8,
    "environment_index": 0,
    "scenario_name": "nominal_stand",
    "command": (0.0, 0.0, 0.20),
    "seed": 20261007,
    "repetition_index": 0,
    "horizon": 499,
}


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonicalize(item) for item in value), key=repr)
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        raise ValueError("Canonical JSON does not permit NaN or Inf")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stable_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def path_is_within(path: str | Path, root: str | Path) -> bool:
    candidate = Path(path).resolve()
    boundary = Path(root).resolve()
    try:
        candidate.relative_to(boundary)
        return True
    except ValueError:
        return False


def require_path_within(
    path: str | Path,
    root: str | Path,
    *,
    label: str,
) -> Path:
    candidate = Path(path).resolve()
    boundary = Path(root).resolve()
    if not path_is_within(candidate, boundary):
        raise ConfigurationError(f"{label} escapes {boundary}: {candidate}")
    return candidate


def validate_run_id(run_id: str) -> str:
    value = str(run_id)
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ConfigurationError(f"Invalid run id: {value!r}")
    return value


def validate_design_identity() -> None:
    if not DESIGN_PATH.is_file():
        raise FileNotFoundError(DESIGN_PATH)
    actual = sha256_file(DESIGN_PATH)
    if actual != DESIGN_SHA256:
        raise IdentityDriftError(
            f"Design SHA256 mismatch: {actual} != {DESIGN_SHA256}"
        )


def require_bootstrap() -> None:
    problems: list[str] = []
    if os.environ.get("WHEELLEG_ROOT_CAUSE_BOOTSTRAP") != "1":
        problems.append("missing bootstrap marker")
    if not sys.flags.dont_write_bytecode:
        problems.append("interpreter was not launched with -B")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        problems.append("PYTHONNOUSERSITE is not 1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        problems.append("PYTHONDONTWRITEBYTECODE is not 1")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not prefix:
        problems.append("PYTHONPYCACHEPREFIX is missing")
    elif SUITE_ROOT.resolve() not in Path(prefix).resolve().parents:
        problems.append("PYTHONPYCACHEPREFIX is outside the suite root")
    if problems:
        raise BootstrapError("Root-cause suite bootstrap contract failed: " + "; ".join(problems))
