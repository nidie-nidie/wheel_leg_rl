from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from .contracts import PROJECT_ROOT, SUITE_ROOT, sha256_file


TEXTURE_CACHE_DEFAULT = "${omni_global_cache}/texturecache"
TEXTURE_CACHE_SETTING = "/rtx-transient/resourcemanager/localTextureCachePath"

FROZEN_ISAAC_SOURCES: Mapping[str, str] = {
    "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/app/app_launcher.py":
        "7AB2C742CFA64E2C4E2EA4601D9441E5E8C9BB8F556A121B0D10C337F9B6093E",
    "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_cfg.py":
        "D8035DF355C576AD7C876789BBA54BE4EBB361F5C16786A4EC8578143EEBC118",
    "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py":
        "3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0",
    "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_context.py":
        "D95F2EAA138B4FFF845AC72439CAA2ECA5DEAB4F0D5609E06F163D20EEF4DF93",
    "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/utils/logger.py":
        "1C800C9876973CED42850AF89FBA68676A920EDA27CDE94E254D454DB32AD63E",
    ".venv/Lib/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/"
    "simulation_app/simulation_app.py":
        "B0C233FD227C57ED38EAFCA2EF728CED31BFD53F444A98682714856CD08716C9",
    ".venv/Lib/site-packages/isaacsim/kit/kernel/config/kit-core.json":
        "A57318200F25E9AB84710C6840398B42B1A4D61AB65CB8561D254D6B3ADB2224",
}


@dataclass(frozen=True)
class KitPaths:
    root: Path
    logs: Path
    data: Path
    cache: Path
    user_config: Path
    crash_dump: Path
    texture_cache: Path

    def payload(self) -> dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def prepare_kit_paths(kit_root: Path, *, run_root: Path) -> KitPaths:
    root = Path(kit_root)
    run = Path(run_root).resolve(strict=True)
    if not root.is_absolute():
        raise ValueError("Kit portable root must be absolute")
    root = root.resolve()
    if not _inside(root, run):
        raise ValueError(f"Kit portable root escapes the run: {root}")
    values = KitPaths(
        root=root,
        logs=root / "logs",
        data=root / "data",
        cache=root / "cache",
        user_config=root / "data" / "user.config.json",
        crash_dump=root / "data",
        texture_cache=root / "cache" / "texturecache",
    )
    for directory in (values.root, values.logs, values.data, values.cache, values.texture_cache):
        directory.mkdir(parents=True, exist_ok=True)
    for value in asdict(values).values():
        path = Path(value).resolve()
        if not path.is_absolute() or not _inside(path, run):
            raise ValueError(f"Resolved Kit path escapes the run: {path}")
    return values


def kit_argument_tokens(paths: KitPaths) -> tuple[str, ...]:
    return (
        "--portable-root",
        str(paths.root),
        "--/crashreporter/enabled=false",
        f"--/crashreporter/dumpDir={paths.crash_dump}",
        f"--/log/file={paths.logs / 'kit.log'}",
        f"--/app/userConfigPath={paths.user_config}",
        f"--{TEXTURE_CACHE_SETTING}={paths.texture_cache}",
    )


def kit_argument_string(paths: KitPaths) -> str:
    tokens = kit_argument_tokens(paths)
    if any(" " in token for token in tokens):
        raise ValueError("Frozen workspace paths must not contain spaces because AppLauncher splits kit_args")
    return " ".join(tokens)


def validate_frozen_isaac_sources() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for relative, expected in FROZEN_ISAAC_SOURCES.items():
        path = PROJECT_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen Isaac source hash mismatch: {relative}: {actual} != {expected}")
        rows[relative] = {"path": str(path.resolve()), "sha256": actual}
    return rows


def validate_kit_core() -> dict[str, Any]:
    relative = ".venv/Lib/site-packages/isaacsim/kit/kernel/config/kit-core.json"
    path = PROJECT_ROOT / relative
    expected = FROZEN_ISAAC_SOURCES[relative]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"kit-core.json hash mismatch: {actual} != {expected}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("rtx-transient", {}).get("resourcemanager", {}).get("localTextureCachePath")
    if value != TEXTURE_CACHE_DEFAULT:
        raise ValueError(f"kit-core texture cache default drifted: {value!r}")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "setting": TEXTURE_CACHE_SETTING,
        "default_value": value,
    }


def _function_calls(source: str, function_name: str) -> list[str]:
    tree = ast.parse(source)
    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None:
        raise ValueError(f"Required function is missing: {function_name}")
    return [ast.unparse(node.func) for node in ast.walk(target) if isinstance(node, ast.Call)]


def validate_source_contracts() -> dict[str, bool]:
    root = PROJECT_ROOT
    app_launcher = (
        root / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/app/app_launcher.py"
    ).read_text(encoding="utf-8")
    simulation_app = (
        root / ".venv/Lib/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/"
        "simulation_app/simulation_app.py"
    ).read_text(encoding="utf-8")
    direct_env = (
        root / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py"
    ).read_text(encoding="utf-8")
    simulation_context = (
        root / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_context.py"
    ).read_text(encoding="utf-8")
    logger = (
        root / "dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/utils/logger.py"
    ).read_text(encoding="utf-8")

    launcher_calls = _function_calls(app_launcher, "_create_app")
    if launcher_calls.count("SimulationApp") != 1 or "self._sim_app_config" not in app_launcher:
        raise ValueError("AppLauncher _create_app contract drifted")
    if 'if self.config["enable_crashreporter"]' not in simulation_app:
        raise ValueError("SimulationApp crash-reporter source contract drifted")
    if "carb.crashreporter-*" not in simulation_app:
        raise ValueError("SimulationApp crash-reporter plugin contract drifted")
    if "self.cfg = cfg" not in direct_env:
        raise ValueError("DirectRLEnv no longer stores cfg before SimulationContext construction")
    if "SimulationContext.instance() is None" not in direct_env:
        raise ValueError("DirectRLEnv existing-context guard drifted")
    if "SimulationContext(self.cfg.sim)" not in direct_env:
        raise ValueError("DirectRLEnv SimulationContext call drifted")
    for token in (
        "logging_level=self.cfg.logging_level",
        "save_logs_to_file=self.cfg.save_logs_to_file",
        "log_dir=self.cfg.log_dir",
    ):
        if token not in simulation_context:
            raise ValueError(f"SimulationContext logger forwarding drifted: {token}")
    if "if log_dir is None:" not in logger or 'tempfile.gettempdir(), "isaaclab", "logs"' not in logger:
        raise ValueError("Isaac Lab logger fallback contract drifted")
    return {
        "app_launcher_single_create": True,
        "crashreporter_boolean_gate": True,
        "direct_env_context_guard": True,
        "simulation_context_logger_forwarding": True,
        "logger_none_only_fallback": True,
    }


LauncherT = TypeVar("LauncherT")


def root_cause_launcher_class(base: type[LauncherT]) -> type[LauncherT]:
    class RootCauseSuiteAppLauncher(base):  # type: ignore[misc, valid-type]
        def _create_app(self) -> Any:
            config = dict(self._sim_app_config)
            config["enable_crashreporter"] = False
            if config.get("enable_crashreporter") is not False:
                raise RuntimeError("Crash reporter must be disabled before SimulationApp construction")
            self._sim_app_config = config
            self._root_cause_prelaunch_config = dict(config)
            return super()._create_app()

    RootCauseSuiteAppLauncher.__name__ = "RootCauseSuiteAppLauncher"
    return RootCauseSuiteAppLauncher


def snapshot_file_handlers() -> list[dict[str, Any]]:
    loggers: list[tuple[str, logging.Logger]] = [("root", logging.getLogger())]
    for name, value in logging.Logger.manager.loggerDict.items():
        if isinstance(value, logging.Logger):
            loggers.append((name, value))
    result: list[dict[str, Any]] = []
    for logger_name, logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                result.append(
                    {
                        "logger": logger_name,
                        "baseFilename": str(Path(handler.baseFilename).resolve()),
                        "mode": handler.mode,
                        "encoding": handler.encoding,
                        "delay": handler.delay,
                    }
                )
    return result


def validate_isaaclab_handlers(
    handlers: list[Mapping[str, Any]], *, expected_directory: Path
) -> list[dict[str, Any]]:
    expected = Path(expected_directory).resolve(strict=True)
    if not handlers:
        raise ValueError("Isaac Lab did not create a FileHandler")
    checked: list[dict[str, Any]] = []
    for handler in handlers:
        filename = Path(str(handler.get("baseFilename"))).resolve(strict=True)
        if filename.parent != expected:
            raise ValueError(f"Isaac Lab FileHandler escaped the exact log directory: {filename}")
        checked.append(dict(handler))
    return checked


def validate_resolved_kit_state(
    paths: KitPaths,
    *,
    resolved: Mapping[str, str],
    crashreporter_enabled: bool,
    loaded_plugins: list[str],
) -> dict[str, Any]:
    exact_expected = {
        "log": (paths.logs / "kit.log").resolve(),
        "config": paths.user_config.resolve(),
        "dump": paths.crash_dump.resolve(),
        "texture_cache": paths.texture_cache.resolve(),
    }
    required = ("log", "data", "cache", "config", "dump", "texture_cache")
    actual = {name: Path(str(resolved[name])).resolve() for name in required}
    escapes = {
        name: str(path)
        for name, path in actual.items()
        if not path.is_absolute() or not _inside(path, paths.root.resolve())
    }
    if escapes:
        raise ValueError(f"Resolved Kit paths escape the portable root: {escapes}")
    mismatches = {
        name: {"expected": str(expected), "actual": str(actual[name])}
        for name, expected in exact_expected.items()
        if actual[name] != expected
    }
    if mismatches:
        raise ValueError(f"Explicit Kit paths differ from requested paths: {mismatches}")
    if crashreporter_enabled:
        raise ValueError("Resolved crash reporter setting is enabled")
    crash_plugins = sorted(name for name in loaded_plugins if name.startswith("carb.crashreporter"))
    if crash_plugins:
        raise ValueError(f"Crash reporter plugins were loaded: {crash_plugins}")
    return {
        "requested": paths.payload(),
        "resolved": {name: str(value) for name, value in actual.items()},
        "crashreporter_enabled": False,
        "loaded_plugins": sorted(loaded_plugins),
        "all_paths_inside_portable_root": True,
        "texture_cache_exact_match": True,
    }
