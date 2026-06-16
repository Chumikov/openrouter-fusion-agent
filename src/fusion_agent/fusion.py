"""Core OpenRouter Fusion execution.

Posts a chat-completion request with the ``openrouter:fusion`` server tool to a
free outer model. OpenRouter runs the panel + judge server-side and returns the
outer model's final answer. The structured ``analysis`` JSON is consumed by the
outer model internally; we surface it best-effort when OpenRouter echoes it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .budget import BudgetTracker, KeyInfo, get_key_info
from .errors import FusionAPIError, FusionBudgetError
from .presets import Preset, pick_panel

logger = logging.getLogger("fusion_agent")

MAX_RETRIES = 3
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
BACKOFF_BASE = 1.6
BACKOFF_MAX = 20.0

# Default per-run cost estimate (free models == $0).
FREE_COST_USD = 0.0


@dataclass
class FusionResult:
    """The outcome of a single fusion run."""

    status: str  # "ok" | "degraded" | "error"
    outer: str
    panel: list[str]
    judge: str
    request_count: int
    final_answer: str | None = None
    analysis: dict[str, Any] | None = None
    panel_responses: list[dict[str, Any]] | None = None
    cost_usd: float | None = None
    failure_reason: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @property
    def ok(self) -> bool:
        return self.status != "error"


def build_payload(
    question: str,
    preset: Preset,
    *,
    force: bool = True,
    panel: tuple[str, ...] | None = None,
    max_completion_tokens: int | None = None,
    max_tool_calls: int | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the chat-completion payload and resolve the concrete panel."""
    panel = panel if panel is not None else pick_panel(preset)
    parameters: dict[str, Any] = {
        "analysis_models": list(panel),
        "model": preset.judge,
    }
    if max_completion_tokens is not None:
        parameters["max_completion_tokens"] = max_completion_tokens
    if max_tool_calls is not None:
        parameters["max_tool_calls"] = max_tool_calls

    payload: dict[str, Any] = {
        "model": preset.outer,
        "messages": [{"role": "user", "content": question}],
        "tools": [{"type": "openrouter:fusion", "parameters": parameters}],
    }
    if force:
        # Guarantee the outer model actually invokes fusion.
        payload["tool_choice"] = "required"
    return payload, panel


def estimate_request_count(panel: tuple[str, ...]) -> int:
    """Approximate OpenRouter completions per run: panel + judge + outer final."""
    return len(panel) + 2


def _probe(data: dict[str, Any], *keys: str) -> Any:
    """Return the first present value among ``keys`` anywhere reasonable."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _find_analysis(data: dict[str, Any], message: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort extraction of the judge's structured analysis JSON."""
    candidate = _probe(data, "analysis", "fusion_analysis")
    if isinstance(candidate, dict):
        return candidate
    candidate = _probe(message, "analysis")
    if isinstance(candidate, dict):
        return candidate
    # Some server tools echo results under tool/message annotations.
    annotations = message.get("annotations") if isinstance(message, dict) else None
    if isinstance(annotations, list):
        for ann in annotations:
            if isinstance(ann, dict) and isinstance(ann.get("analysis"), dict):
                return ann["analysis"]  # type: ignore[no-any-return]
    return None


def _find_panel_responses(
    data: dict[str, Any], message: dict[str, Any]
) -> list[dict[str, Any]] | None:
    candidate = _probe(data, "responses", "panel_responses")
    if isinstance(candidate, list):
        return candidate
    candidate = _probe(message, "responses")
    if isinstance(candidate, list):
        return candidate
    return None


def _find_failure_reason(data: dict[str, Any], message: dict[str, Any]) -> str | None:
    candidate = _probe(data, "failure_reason") or _probe(message, "failure_reason")
    return str(candidate) if candidate else None


def parse_completion(
    data: dict[str, Any],
    *,
    panel: tuple[str, ...],
    preset: Preset,
    request_count: int,
) -> FusionResult:
    """Turn an OpenRouter chat-completion body into a ``FusionResult``."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    final_answer = message.get("content")
    outer_model = str(data.get("model") or preset.outer)

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    cost = usage.get("cost") if isinstance(usage, dict) else None
    cost_usd = float(cost) if isinstance(cost, int | float) else None

    analysis = _find_analysis(data, message)
    panel_responses = _find_panel_responses(data, message)
    failure_reason = _find_failure_reason(data, message)

    if failure_reason:
        status = "error"
    elif final_answer:
        status = "ok"
    else:
        status = "degraded"

    return FusionResult(
        status=status,
        outer=outer_model,
        panel=list(panel),
        judge=preset.judge,
        request_count=request_count,
        final_answer=str(final_answer) if final_answer else None,
        analysis=analysis,
        panel_responses=panel_responses,
        cost_usd=cost_usd,
        failure_reason=failure_reason,
        raw=data,
    )


async def _post_with_retry(client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("network error on fusion attempt %d: %s", attempt, exc)
            await _backoff(attempt)
            continue

        if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
            logger.warning("retryable HTTP %d on fusion attempt %d", response.status_code, attempt)
            # 402 (insufficient credits / negative balance) is not recoverable
            # by waiting, but the panel may still degrade gracefully; retry once.
            await _backoff(attempt)
            continue

        if response.status_code >= 400:
            raise FusionAPIError(
                f"OpenRouter returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
            )

        return response.json()  # type: ignore[no-any-return]

    raise FusionAPIError(
        f"fusion request failed after {MAX_RETRIES + 1} attempts: {last_error}",
        status_code=None,
    )


async def _backoff(attempt: int) -> None:
    delay = min(BACKOFF_BASE**attempt, BACKOFF_MAX)
    await asyncio.sleep(delay)


async def run_fusion(
    client: httpx.AsyncClient,
    question: str,
    preset: Preset,
    *,
    force: bool = True,
    panel_size: int | None = None,
    tracker: BudgetTracker | None = None,
    max_completion_tokens: int | None = None,
    max_tool_calls: int | None = None,
) -> FusionResult:
    """Execute a single fusion deliberation and return a ``FusionResult``."""
    panel = pick_panel(preset, panel_size)
    request_count = estimate_request_count(panel)

    if tracker is not None:
        await tracker.throttle_rpm(request_count)
        if not tracker.can_run(request_count):
            raise FusionBudgetError(
                f"daily free-model budget exhausted: {tracker.used_today}/{tracker.rpd_cap} "
                f"requests used, need {request_count} more."
            )

    payload, panel = build_payload(
        question,
        preset,
        force=force,
        panel=panel,
        max_completion_tokens=max_completion_tokens,
        max_tool_calls=max_tool_calls,
    )

    try:
        data = await _post_with_retry(client, payload)
    finally:
        if tracker is not None:
            tracker.record(request_count)

    result = parse_completion(data, panel=panel, preset=preset, request_count=request_count)
    if result.cost_usd is None:
        result.cost_usd = FREE_COST_USD
    return result


__all__ = [
    "FusionResult",
    "KeyInfo",
    "build_payload",
    "estimate_request_count",
    "get_key_info",
    "parse_completion",
    "run_fusion",
]
