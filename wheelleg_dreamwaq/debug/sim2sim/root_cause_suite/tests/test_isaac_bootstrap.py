from __future__ import annotations

from pathlib import Path

import pytest

from debug.sim2sim.root_cause_suite.isaac_bootstrap import (
    TEXTURE_CACHE_SETTING,
    kit_argument_tokens,
    prepare_kit_paths,
    root_cause_launcher_class,
    validate_frozen_isaac_sources,
    validate_isaaclab_handlers,
    validate_kit_core,
    validate_resolved_kit_state,
    validate_source_contracts,
)


class _FakeLauncher:
    def __init__(self) -> None:
        self._sim_app_config = {"enable_crashreporter": True, "headless": True}
        self.calls = 0

    def _create_app(self) -> str:
        self.calls += 1
        assert self._sim_app_config["enable_crashreporter"] is False
        return "created"


def test_frozen_sources_and_source_contracts_match_workspace() -> None:
    assert validate_frozen_isaac_sources()
    assert validate_source_contracts()["crashreporter_boolean_gate"] is True
    assert validate_kit_core()["default_value"] == "${omni_global_cache}/texturecache"


def test_launcher_forces_crashreporter_false_before_one_super_call() -> None:
    launcher_type = root_cause_launcher_class(_FakeLauncher)
    launcher = launcher_type()
    assert launcher._create_app() == "created"
    assert launcher.calls == 1
    assert launcher._root_cause_prelaunch_config["enable_crashreporter"] is False


def test_kit_paths_are_absolute_exact_and_run_local(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    paths = prepare_kit_paths((run / "runtime_cache" / "kit").resolve(), run_root=run)
    tokens = kit_argument_tokens(paths)
    assert tokens[:2] == ("--portable-root", str(paths.root))
    assert f"--{TEXTURE_CACHE_SETTING}={paths.texture_cache}" in tokens
    resolved = {
        "log": str(paths.logs / "kit.log"),
        "data": str(paths.data / "Kit" / "Isaac-Sim" / "5.1"),
        "cache": str(paths.cache / "Kit" / "Isaac-Sim" / "5.1"),
        "config": str(paths.user_config),
        "dump": str(paths.crash_dump),
        "texture_cache": str(paths.texture_cache),
    }
    assert validate_resolved_kit_state(
        paths, resolved=resolved, crashreporter_enabled=False, loaded_plugins=["carb.core"]
    )["texture_cache_exact_match"] is True


def test_kit_path_and_resolved_sibling_escapes_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        prepare_kit_paths((tmp_path / "outside").resolve(), run_root=run)
    paths = prepare_kit_paths((run / "runtime_cache" / "kit").resolve(), run_root=run)
    resolved = {
        "log": str(paths.logs / "kit.log"),
        "data": str(paths.data),
        "cache": str(paths.cache),
        "config": str(paths.user_config),
        "dump": str(paths.crash_dump),
        "texture_cache": str(paths.cache / "wrong-sibling"),
    }
    with pytest.raises(ValueError, match="differ"):
        validate_resolved_kit_state(
            paths, resolved=resolved, crashreporter_enabled=False, loaded_plugins=[]
        )


def test_handler_parent_must_match_exact_log_directory(tmp_path: Path) -> None:
    expected = tmp_path / "logs"
    expected.mkdir()
    legal = expected / "isaaclab.log"
    legal.write_text("", encoding="utf-8")
    assert validate_isaaclab_handlers(
        [{"baseFilename": str(legal)}], expected_directory=expected
    )
    sibling = tmp_path / "other"
    sibling.mkdir()
    escaped = sibling / "isaaclab.log"
    escaped.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="escaped"):
        validate_isaaclab_handlers(
            [{"baseFilename": str(escaped)}], expected_directory=expected
        )
