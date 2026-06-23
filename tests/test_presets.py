from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_agent.presets import (
    DEFAULT_CONFIG,
    get_config,
    panel_candidates,
    pick_panel,
    reload_config,
)

# --- ModelConfig properties -------------------------------------------------


def test_default_config_has_candidates() -> None:
    assert len(DEFAULT_CONFIG.outer) >= 2
    assert len(DEFAULT_CONFIG.judge) >= 2
    assert len(DEFAULT_CONFIG.panel) >= 1
    assert DEFAULT_CONFIG.primary_outer == DEFAULT_CONFIG.outer[0]
    assert DEFAULT_CONFIG.primary_judge == DEFAULT_CONFIG.judge[0]


def test_default_config_all_free() -> None:
    for model in DEFAULT_CONFIG.outer:
        assert model.endswith(":free")
    for model in DEFAULT_CONFIG.judge:
        assert model.endswith(":free")
    for panel in DEFAULT_CONFIG.panel:
        for model in panel:
            assert model.endswith(":free")


# --- pick_panel -------------------------------------------------------------


def test_pick_panel_default() -> None:
    assert pick_panel(DEFAULT_CONFIG) == DEFAULT_CONFIG.primary_panel


@pytest.mark.parametrize("size", [1, 2, 3])
def test_pick_panel_shrinks(size: int) -> None:
    assert pick_panel(DEFAULT_CONFIG, size) == DEFAULT_CONFIG.primary_panel[:size]


def test_pick_panel_invalid_size() -> None:
    with pytest.raises(ValueError, match="panel_size"):
        pick_panel(DEFAULT_CONFIG, 0)
    primary_len = len(DEFAULT_CONFIG.primary_panel)
    with pytest.raises(ValueError, match="panel_size"):
        pick_panel(DEFAULT_CONFIG, primary_len + 1)


# --- panel_candidates -------------------------------------------------------


def test_panel_candidates_returns_all() -> None:
    candidates = panel_candidates(DEFAULT_CONFIG)
    assert len(candidates) == len(DEFAULT_CONFIG.panel)
    assert candidates[0] == DEFAULT_CONFIG.primary_panel


def test_panel_candidates_trimmed() -> None:
    candidates = panel_candidates(DEFAULT_CONFIG, size=2)
    for panel in candidates:
        assert len(panel) == 2


# --- get_config / file loading ----------------------------------------------


def test_get_config_fallback_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Point to a nonexistent file → should fall back to DEFAULT_CONFIG.
    reload_config()
    monkeypatch.setenv("FUSION_MODELS_FILE", str(tmp_path / "nonexistent.json"))
    config = get_config()
    assert config == DEFAULT_CONFIG


def test_get_config_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_config = {
        "version": 1,
        "outer": ["custom/outer-1:free", "custom/outer-2:free"],
        "panel": [["custom/p1:free", "custom/p2:free", "custom/p3:free"]],
        "judge": ["custom/judge:free"],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(file_config), encoding="utf-8")

    reload_config()
    monkeypatch.setenv("FUSION_MODELS_FILE", str(path))
    config = get_config()
    assert config.primary_outer == "custom/outer-1:free"
    assert config.outer == ("custom/outer-1:free", "custom/outer-2:free")
    assert config.primary_judge == "custom/judge:free"
    assert config.primary_panel == ("custom/p1:free", "custom/p2:free", "custom/p3:free")


def test_reload_config_invalidates_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"outer": ["first:free"], "panel": [["p:free"]], "judge": ["j:free"]}),
        encoding="utf-8",
    )
    reload_config()
    monkeypatch.setenv("FUSION_MODELS_FILE", str(path))
    assert get_config().primary_outer == "first:free"

    # Overwrite file and reload.
    path.write_text(
        json.dumps({"outer": ["second:free"], "panel": [["p:free"]], "judge": ["j:free"]}),
        encoding="utf-8",
    )
    reload_config()
    assert get_config().primary_outer == "second:free"


def test_get_config_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"outer": ["cached:free"], "panel": [["p:free"]], "judge": ["j:free"]}),
        encoding="utf-8",
    )
    reload_config()
    monkeypatch.setenv("FUSION_MODELS_FILE", str(path))
    first = get_config()
    # Overwrite but don't reload → should return cached.
    path.write_text(
        json.dumps({"outer": ["changed:free"], "panel": [["p:free"]], "judge": ["j:free"]}),
        encoding="utf-8",
    )
    second = get_config()
    assert first is second
    assert second.primary_outer == "cached:free"
