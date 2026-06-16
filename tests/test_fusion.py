from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
import respx

from fusion_agent.budget import BudgetTracker
from fusion_agent.errors import FusionAPIError, FusionBudgetError
from fusion_agent.fusion import (
    build_payload,
    estimate_request_count,
    parse_completion,
    run_fusion,
)
from fusion_agent.http import build_client
from fusion_agent.presets import QUALITY

from .conftest import sample_completion, sample_key_info

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest_asyncio.fixture
async def client(api_key: str) -> AsyncIterator[httpx.AsyncClient]:
    c = build_client()
    try:
        yield c
    finally:
        await c.aclose()


# --- pure-function tests -----------------------------------------------------


def test_build_payload_force_adds_tool_choice() -> None:
    payload, panel = build_payload("hi", QUALITY, force=True)
    assert payload["model"] == QUALITY.outer
    assert payload["tool_choice"] == "required"
    tool = payload["tools"][0]
    assert tool["type"] == "openrouter:fusion"
    assert tool["parameters"]["analysis_models"] == list(panel)
    assert tool["parameters"]["model"] == QUALITY.judge


def test_build_payload_no_force_omits_tool_choice() -> None:
    payload, _ = build_payload("hi", QUALITY, force=False)
    assert "tool_choice" not in payload


def test_build_payload_respects_panel_size() -> None:
    payload, panel = build_payload("hi", QUALITY, force=True, panel=QUALITY.panel[:2])
    assert panel == QUALITY.panel[:2]
    assert len(payload["tools"][0]["parameters"]["analysis_models"]) == 2


def test_estimate_request_count() -> None:
    assert estimate_request_count(QUALITY.panel) == 5
    assert estimate_request_count(QUALITY.panel[:2]) == 4


def test_parse_completion_ok_with_analysis() -> None:
    analysis = {"consensus": ["x"], "contradictions": [], "blind_spots": ["y"]}
    data = sample_completion(answer="answer", analysis=analysis)
    result = parse_completion(data, panel=QUALITY.panel, preset=QUALITY, request_count=5)
    assert result.status == "ok"
    assert result.final_answer == "answer"
    assert result.analysis == analysis
    assert result.cost_usd == 0.0
    assert result.outer == "qwen/qwen3-next-80b-a3b-instruct:free"


def test_parse_completion_degraded_without_answer() -> None:
    data = sample_completion(answer="")
    result = parse_completion(data, panel=QUALITY.panel, preset=QUALITY, request_count=5)
    assert result.status == "degraded"
    assert result.final_answer is None


def test_parse_completion_error_with_failure_reason() -> None:
    data = sample_completion()
    data["failure_reason"] = "all_panels_failed"
    result = parse_completion(data, panel=QUALITY.panel, preset=QUALITY, request_count=5)
    assert result.status == "error"
    assert result.failure_reason == "all_panels_failed"


def test_parse_completion_missing_cost_defaults() -> None:
    data = sample_completion()
    data.pop("usage", None)
    result = parse_completion(data, panel=QUALITY.panel, preset=QUALITY, request_count=5)
    # run_fusion fills cost; parse alone leaves it None.
    assert result.cost_usd is None


# --- network tests (mocked) --------------------------------------------------


@respx.mock
async def test_run_fusion_success(client: httpx.AsyncClient) -> None:
    respx.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(200, json=sample_key_info())
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=sample_completion(answer="42"))
    )
    result = await run_fusion(
        client, "meaning of life?", QUALITY, tracker=BudgetTracker(rpd_cap=1000)
    )
    assert route.called
    assert result.ok
    assert result.final_answer == "42"
    assert result.request_count == 5


async def test_run_fusion_budget_exhausted_raises(client: httpx.AsyncClient) -> None:
    tracker = BudgetTracker(rpd_cap=0)
    with pytest.raises(FusionBudgetError, match="budget exhausted"):
        await run_fusion(client, "q", QUALITY, tracker=tracker)


@respx.mock
async def test_run_fusion_retries_on_429_then_succeeds(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Avoid real backoff sleeps during the retry.
    async def _fast(_attempt: int) -> None:
        return None

    monkeypatch.setattr("fusion_agent.fusion._backoff", _fast)

    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=sample_completion(answer="recovered")),
        ]
    )
    result = await run_fusion(client, "q", QUALITY, tracker=BudgetTracker(rpd_cap=1000))
    assert route.call_count == 2
    assert result.final_answer == "recovered"


@respx.mock
async def test_run_fusion_raises_on_persistent_server_error(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fast(_attempt: int) -> None:
        return None

    monkeypatch.setattr("fusion_agent.fusion._backoff", _fast)

    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(FusionAPIError, match="HTTP 500"):
        await run_fusion(client, "q", QUALITY, tracker=BudgetTracker(rpd_cap=1000))


@respx.mock
async def test_run_fusion_records_usage_in_tracker(client: httpx.AsyncClient) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=sample_completion()))
    tracker = BudgetTracker(rpd_cap=1000)
    await run_fusion(client, "q", QUALITY, tracker=tracker)
    assert tracker.used_today == 5
