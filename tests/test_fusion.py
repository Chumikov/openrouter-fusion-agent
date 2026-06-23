from __future__ import annotations

import json
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
    probe_model,
    run_fusion,
)
from fusion_agent.http import build_client
from fusion_agent.presets import DEFAULT_CONFIG

from .conftest import (
    sample_completion,
    sample_completion_midstream_error,
    sample_error_response,
    sample_key_info,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
CONFIG = DEFAULT_CONFIG


@pytest_asyncio.fixture
async def client(api_key: str) -> AsyncIterator[httpx.AsyncClient]:
    c = build_client()
    try:
        yield c
    finally:
        await c.aclose()


def _request_model(request: httpx.Request) -> str:
    """Extract the 'model' field from a mocked request body."""
    return str(json.loads(request.content)["model"])


# --- pure-function tests ----------------------------------------------------


def test_build_payload_force_adds_tool_choice() -> None:
    payload, panel = build_payload("hi", CONFIG, force=True)
    assert payload["model"] == CONFIG.primary_outer
    assert payload["tool_choice"] == "required"
    tool = payload["tools"][0]
    assert tool["type"] == "openrouter:fusion"
    assert tool["parameters"]["analysis_models"] == list(panel)
    assert tool["parameters"]["model"] == CONFIG.primary_judge


def test_build_payload_no_force_omits_tool_choice() -> None:
    payload, _ = build_payload("hi", CONFIG, force=False)
    assert "tool_choice" not in payload


def test_build_payload_outer_judge_overrides() -> None:
    payload, _ = build_payload(
        "hi", CONFIG, force=True, outer="custom/outer:free", judge="custom/judge:free"
    )
    assert payload["model"] == "custom/outer:free"
    assert payload["tools"][0]["parameters"]["model"] == "custom/judge:free"


def test_build_payload_respects_panel_size() -> None:
    payload, panel = build_payload("hi", CONFIG, force=True, panel=CONFIG.primary_panel[:2])
    assert panel == CONFIG.primary_panel[:2]
    assert len(payload["tools"][0]["parameters"]["analysis_models"]) == 2


def test_estimate_request_count() -> None:
    assert estimate_request_count(CONFIG.primary_panel) == 5
    assert estimate_request_count(CONFIG.primary_panel[:2]) == 4


def test_parse_completion_ok_with_analysis() -> None:
    analysis = {"consensus": ["x"], "contradictions": [], "blind_spots": ["y"]}
    data = sample_completion(answer="answer", analysis=analysis)
    result = parse_completion(data, panel=CONFIG.primary_panel, config=CONFIG, request_count=5)
    assert result.status == "ok"
    assert result.final_answer == "answer"
    assert result.analysis == analysis
    assert result.cost_usd == 0.0


def test_parse_completion_degraded_without_answer() -> None:
    data = sample_completion(answer="")
    result = parse_completion(data, panel=CONFIG.primary_panel, config=CONFIG, request_count=5)
    assert result.status == "degraded"
    assert result.final_answer is None


def test_parse_completion_error_with_failure_reason() -> None:
    data = sample_completion()
    data["failure_reason"] = "all_panels_failed"
    result = parse_completion(data, panel=CONFIG.primary_panel, config=CONFIG, request_count=5)
    assert result.status == "error"
    assert result.failure_reason == "all_panels_failed"


def test_parse_completion_midstream_error() -> None:
    data = sample_completion_midstream_error(message="Provider overloaded")
    result = parse_completion(data, panel=CONFIG.primary_panel, config=CONFIG, request_count=5)
    assert result.status == "error"
    assert "Provider overloaded" in (result.failure_reason or "")


def test_parse_completion_missing_cost_defaults() -> None:
    data = sample_completion()
    data.pop("usage", None)
    result = parse_completion(data, panel=CONFIG.primary_panel, config=CONFIG, request_count=5)
    assert result.cost_usd is None


# --- probe_model ------------------------------------------------------------


@respx.mock
async def test_probe_model_true_on_200(client: httpx.AsyncClient) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=sample_completion()))
    assert await probe_model(client, "openai/gpt-oss-120b:free") is True


@respx.mock
async def test_probe_model_false_on_429(client: httpx.AsyncClient) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(429, json=sample_error_response()))
    assert await probe_model(client, "qwen/qwen3-next-80b-a3b-instruct:free") is False


# --- network tests (mocked) -------------------------------------------------


@respx.mock
async def test_run_fusion_success(client: httpx.AsyncClient) -> None:
    respx.get("https://openrouter.ai/api/v1/key").mock(
        return_value=httpx.Response(200, json=sample_key_info())
    )
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(200, json=sample_completion(answer="42"))
    )
    result = await run_fusion(
        client, "meaning of life?", CONFIG, tracker=BudgetTracker(rpd_cap=1000)
    )
    assert route.called
    assert result.ok
    assert result.final_answer == "42"
    assert result.request_count == 5


async def test_run_fusion_budget_exhausted_raises(client: httpx.AsyncClient) -> None:
    tracker = BudgetTracker(rpd_cap=0)
    with pytest.raises(FusionBudgetError, match="budget exhausted"):
        await run_fusion(client, "q", CONFIG, tracker=tracker)


def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fast(_attempt: int, **_kw: object) -> None:
        return None

    monkeypatch.setattr("fusion_agent.fusion._backoff", _fast)


@respx.mock
async def test_run_fusion_retries_on_429_then_succeeds(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_backoff(monkeypatch)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json=sample_error_response()),
            httpx.Response(200, json=sample_completion(answer="recovered")),
        ]
    )
    result = await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))
    assert route.call_count == 2
    assert result.final_answer == "recovered"


@respx.mock
async def test_run_fusion_raises_on_persistent_server_error(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_backoff(monkeypatch)
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(FusionAPIError, match="HTTP 500"):
        await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))


@respx.mock
async def test_run_fusion_records_usage_in_tracker(client: httpx.AsyncClient) -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=sample_completion()))
    tracker = BudgetTracker(rpd_cap=1000)
    await run_fusion(client, "q", CONFIG, tracker=tracker)
    assert tracker.used_today == 5


# --- model rotation tests ---------------------------------------------------


@respx.mock
async def test_run_fusion_rotates_outer_on_429(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary outer 429s → probe backup → fusion succeeds with backup."""
    _no_backoff(monkeypatch)
    primary = CONFIG.primary_outer
    backup = CONFIG.outer[1]

    def handler(request: httpx.Request) -> httpx.Response:
        if _request_model(request) == primary:
            return httpx.Response(429, json=sample_error_response())
        return httpx.Response(200, json=sample_completion(answer="from backup", model=backup))

    respx.post(CHAT_URL).mock(side_effect=handler)
    result = await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))
    assert result.ok
    assert result.final_answer == "from backup"
    assert result.outer == backup
    assert result.models_tried is not None
    assert any(primary in entry for entry in result.models_tried)


@respx.mock
async def test_run_fusion_rotates_on_503_error_type(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 with error_type=provider_overloaded triggers rotation."""
    _no_backoff(monkeypatch)
    primary = CONFIG.primary_outer
    backup = CONFIG.outer[1]

    def handler(request: httpx.Request) -> httpx.Response:
        if _request_model(request) == primary:
            return httpx.Response(
                503, json=sample_error_response(code=503, error_type="provider_overloaded")
            )
        return httpx.Response(200, json=sample_completion(model=backup))

    respx.post(CHAT_URL).mock(side_effect=handler)
    result = await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))
    assert result.ok
    assert result.outer == backup


@respx.mock
async def test_run_fusion_no_rotate_on_402(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """402 (payment_required) does not rotate — fails immediately."""
    _no_backoff(monkeypatch)
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            402, json=sample_error_response(code=402, error_type="payment_required")
        )
    )
    with pytest.raises(FusionAPIError, match="HTTP 402"):
        await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))


@respx.mock
async def test_run_fusion_all_candidates_exhausted(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All outer candidates unavailable → FusionAPIError."""
    _no_backoff(monkeypatch)

    async def _probe_false(_client: httpx.AsyncClient, _model: str) -> bool:
        return False

    monkeypatch.setattr("fusion_agent.fusion.probe_model", _probe_false)
    respx.post(CHAT_URL).mock(return_value=httpx.Response(429, json=sample_error_response()))
    with pytest.raises(FusionAPIError, match="all model candidates exhausted"):
        await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))


@respx.mock
async def test_run_fusion_retry_after_short_retries_same_model(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry-After <= 10s → wait and retry the same model."""
    delays: list[float] = []

    async def _track(attempt: int, *, retry_after: float | None = None) -> None:
        if retry_after is not None:
            delays.append(retry_after)

    monkeypatch.setattr("fusion_agent.fusion._backoff", _track)

    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                429,
                json=sample_error_response(),
                headers={"Retry-After": "3"},
            ),
            httpx.Response(200, json=sample_completion(answer="waited")),
        ]
    )
    result = await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))
    assert route.call_count == 2
    assert result.final_answer == "waited"
    assert 3.0 in delays


@respx.mock
async def test_run_fusion_invalid_json_retries_then_rotates(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """200 OK with invalid JSON body → retry → if persistent, rotate (treated as 502)."""
    _no_backoff(monkeypatch)
    primary = CONFIG.primary_outer
    backup = CONFIG.outer[1]

    def handler(request: httpx.Request) -> httpx.Response:
        if _request_model(request) == primary:
            # 200 OK but body is not valid JSON.
            return httpx.Response(200, text="<html>Gateway Error</html>")
        return httpx.Response(200, json=sample_completion(model=backup))

    respx.post(CHAT_URL).mock(side_effect=handler)
    result = await run_fusion(client, "q", CONFIG, tracker=BudgetTracker(rpd_cap=1000))
    assert result.ok
    assert result.outer == backup
    assert result.models_tried is not None
