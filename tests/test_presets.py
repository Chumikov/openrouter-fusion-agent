from __future__ import annotations

import pytest

from fusion_agent.presets import BUDGET, QUALITY, get_preset, pick_panel


def test_get_preset_known() -> None:
    assert get_preset("quality") is QUALITY
    assert get_preset("BUDGET") is BUDGET  # case-insensitive


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("fast")


def test_pick_panel_default_returns_full() -> None:
    assert pick_panel(QUALITY) == QUALITY.panel


@pytest.mark.parametrize("size", [1, 2, 3])
def test_pick_panel_shrinks(size: int) -> None:
    assert pick_panel(QUALITY, size) == QUALITY.panel[:size]


def test_pick_panel_invalid_size() -> None:
    with pytest.raises(ValueError, match="panel_size"):
        pick_panel(QUALITY, 0)
    with pytest.raises(ValueError, match="panel_size"):
        pick_panel(QUALITY, 4)


def test_quality_panel_is_family_diverse() -> None:
    # Families (top-level vendor) should be distinct for less correlated answers.
    families = {model.split("/")[0] for model in QUALITY.panel}
    assert families == {"openai", "nvidia", "meta-llama"}


def test_judge_and_outer_present() -> None:
    for preset in (QUALITY, BUDGET):
        assert preset.outer
        assert preset.judge
        assert preset.judge not in preset.panel or preset.panel.count(preset.judge) <= 1


def test_all_models_are_free_variants() -> None:
    for preset in (QUALITY, BUDGET):
        models = {preset.outer, preset.judge, *preset.panel}
        for model in models:
            assert model.endswith(":free"), model
