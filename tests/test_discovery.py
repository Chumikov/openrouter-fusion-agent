from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx

from fusion_agent.discovery import (
    ModelInfo,
    fetch_free_tool_models,
    load_selection,
    parse_param_count,
    pick_diverse,
    save_selection,
    select_models,
)
from fusion_agent.http import build_client

from .conftest import sample_models_catalog

MODELS_URL = "https://openrouter.ai/api/v1/models"


@pytest_asyncio.fixture
async def client(api_key: str) -> AsyncIterator[httpx.AsyncClient]:
    c = build_client()
    try:
        yield c
    finally:
        await c.aclose()


# --- parse_param_count ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("openai/gpt-oss-120b:free", 120.0),
        ("nvidia/nemotron-3-ultra-550b-a55b:free", 550.0),
        ("qwen/qwen3-next-80b-a3b-instruct:free", 80.0),
        ("meta-llama/llama-3.3-70b-instruct:free", 70.0),
        ("google/gemma-4-26b-a4b-it:free", 26.0),
        ("liquid/lfm-2.5-1.2b-thinking:free", 1.2),
        ("cohere/north-mini-code:free", None),
        ("qwen/qwen3-coder:free", None),
    ],
)
def test_parse_param_count(text: str, expected: float | None) -> None:
    assert parse_param_count(text) == expected


def test_parse_param_count_multiple_texts() -> None:
    # ID has no size, name does — should pick up from name.
    assert parse_param_count("qwen/qwen3-coder:free", "Qwen3 Coder 480B A35B") == 480.0


def test_parse_param_count_takes_largest() -> None:
    assert parse_param_count("model-550b-a55b") == 550.0


# --- fetch_free_tool_models -------------------------------------------------


@respx.mock
async def test_fetch_free_tool_models(client: httpx.AsyncClient) -> None:
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=sample_models_catalog()))
    models = await fetch_free_tool_models(client)
    ids = [m.id for m in models]
    # 8 entries in catalog, 1 lacks tools → 7 remain.
    assert len(models) == 7
    assert "cohere/north-mini-code:free" not in ids
    assert all(m.id.endswith(":free") for m in models)


@respx.mock
async def test_fetch_free_tool_models_enriches(client: httpx.AsyncClient) -> None:
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=sample_models_catalog()))
    models = await fetch_free_tool_models(client)
    by_id = {m.id: m for m in models}
    gpt = by_id["openai/gpt-oss-120b:free"]
    assert gpt.family == "openai"
    assert gpt.param_count == 120.0
    assert gpt.context_length == 131072


# --- select_models ----------------------------------------------------------


def _make_models() -> list[ModelInfo]:
    return [
        ModelInfo("qwen/qwen3-next-80b-a3b-instruct:free", "qwen", 80.0, 262144, "Qwen3"),
        ModelInfo("openai/gpt-oss-120b:free", "openai", 120.0, 131072, "gpt-oss"),
        ModelInfo("nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia", 550.0, 1000000, "Ultra"),
        ModelInfo("meta-llama/llama-3.3-70b-instruct:free", "meta-llama", 70.0, 131072, "Llama"),
        ModelInfo("google/gemma-4-26b-a4b-it:free", "google", 26.0, 262144, "Gemma"),
        ModelInfo("openai/gpt-oss-20b:free", "openai", 20.0, 131072, "gpt-oss-20"),
        ModelInfo("liquid/lfm-2.5-1.2b-thinking:free", "liquid", 1.2, 32768, "LFM"),
    ]


def test_select_models_outer_diverse_families() -> None:
    result = select_models(_make_models(), min_b=20)
    outer_families = {m.split("/")[0] for m in result["outer"]}
    # 3 outer models from 3 different families.
    assert len(result["outer"]) == 3
    assert len(outer_families) == 3


def test_select_models_min_b_threshold() -> None:
    result = select_models(_make_models(), min_b=20)
    for role in ("outer", "judge"):
        for model_id in result[role]:  # type: ignore[union-attr]
            # All selected models should be >= 20B (the 1.2B liquid model excluded).
            assert "lfm-2.5" not in model_id


def test_select_models_judge_not_outer_family() -> None:
    result = select_models(_make_models(), min_b=20)
    outer_family = result["outer"][0].split("/")[0]  # type: ignore[index]
    judge_families = {m.split("/")[0] for m in result["judge"]}  # type: ignore[union-attr]
    assert outer_family not in judge_families


def test_select_models_panel_has_three_diverse() -> None:
    result = select_models(_make_models(), min_b=20)
    panel = result["panel"][0]  # type: ignore[index]
    assert len(panel) == 3
    panel_families = {m.split("/")[0] for m in panel}
    assert len(panel_families) == 3


def test_select_models_generates_backup_panel() -> None:
    """With enough models, select_models should generate >= 2 panel compositions."""
    result = select_models(_make_models(), min_b=20)
    panels = result["panel"]  # type: ignore[index]
    assert len(panels) >= 2, f"expected >= 2 panel compositions, got {len(panels)}"
    # Each composition must have >= 2 models.
    for panel in panels:
        assert len(panel) >= 2, f"panel composition too small: {panel}"


def test_select_models_all_panels_meet_minimum() -> None:
    """No panel composition should have fewer than 2 models."""
    result = select_models(_make_models(), min_b=20)
    for panel in result["panel"]:  # type: ignore[index]
        assert len(panel) >= 2


def test_select_models_relaxes_threshold_when_few() -> None:
    # Only models below min_b.
    small = [
        ModelInfo("liquid/lfm-2.5-1.2b:free", "liquid", 1.2, 32768, "LFM"),
        ModelInfo("other/small-3b:free", "other", 3.0, 32768, "Small"),
    ]
    result = select_models(small, min_b=20)
    # Should not return empty — relaxes threshold.
    assert len(result["outer"]) >= 1  # type: ignore[arg-type]


def test_select_models_all_free() -> None:
    result = select_models(_make_models(), min_b=20)
    for role in ("outer", "judge"):
        for model_id in result[role]:  # type: ignore[union-attr]
            assert model_id.endswith(":free")


# --- pick_diverse -----------------------------------------------------------


def test_pick_diverse_prefers_distinct_families() -> None:
    pool = _make_models()
    picked = pick_diverse(pool, 3)
    families = {m.family for m in picked}
    assert len(families) == 3


def test_pick_diverse_fills_when_few_families() -> None:
    pool = [
        ModelInfo("a/m-1:free", "a", 10, 1000, ""),
        ModelInfo("a/m-2:free", "a", 20, 1000, ""),
    ]
    picked = pick_diverse(pool, 3)
    assert len(picked) == 2  # Can't pick more than available.


# --- save / load roundtrip --------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    selection = {
        "outer": ["a:free", "b:free"],
        "panel": [["c:free", "d:free", "e:free"]],
        "judge": ["f:free"],
    }
    path = save_selection(tmp_path / "models.json", selection, total_found=7)
    assert path.exists()

    loaded = load_selection(path)
    assert loaded is not None
    assert loaded["outer"] == ["a:free", "b:free"]  # type: ignore[index]
    assert loaded["judge"] == ["f:free"]  # type: ignore[index]
    assert loaded["total_free_tool_models"] == 7  # type: ignore[index]


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_selection(tmp_path / "nonexistent.json") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_selection(path) is None


def test_load_missing_keys_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"outer": ["a:free"]}), encoding="utf-8")
    assert load_selection(path) is None
