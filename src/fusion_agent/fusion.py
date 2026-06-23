"""Core OpenRouter Fusion execution with automatic model rotation.

Posts a chat-completion request with the ``openrouter:fusion`` server tool to a
free outer model. OpenRouter runs the panel + judge server-side and returns the
outer model's final answer. When a model is unavailable (HTTP 429/502/503/504),
the agent automatically rotates to the next backup model from the priority list
before giving up.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .budget import BudgetTracker, KeyInfo, get_key_info
from .errors import FusionAPIError, FusionBudgetError
from .presets import ModelConfig, pick_panel

logger = logging.getLogger("fusion_agent")

MAX_RETRIES = 3
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Status codes where switching to a *different* model may help (per OpenRouter
# docs: "Rate Limiting and Availability" category).
ROTATE_STATUS = {429, 502, 503, 504}
# Typed error_type values (from error.metadata.error_type) that trigger rotation.
ROTATE_ERROR_TYPES = {"rate_limit_exceeded", "provider_overloaded", "provider_unavailable"}

BACKOFF_BASE = 1.6
BACKOFF_MAX = 20.0
RETRY_AFTER_SHORT = 10.0  # seconds; if Retry-After <= this, wait + retry same model

PROBE_TIMEOUT = 15.0

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
    models_tried: list[str] | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.status != "error"


def build_payload(
    question: str,
    config: ModelConfig,
    *,
    force: bool = True,
    panel: tuple[str, ...] | None = None,
    outer: str | None = None,
    judge: str | None = None,
    max_completion_tokens: int | None = None,
    max_tool_calls: int | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the chat-completion payload and resolve the concrete panel."""
    panel = panel if panel is not None else pick_panel(config)
    outer_model = outer if outer is not None else config.primary_outer
    judge_model = judge if judge is not None else config.primary_judge

    parameters: dict[str, Any] = {
        "analysis_models": list(panel),
        "model": judge_model,
    }
    if max_completion_tokens is not None:
        parameters["max_completion_tokens"] = max_completion_tokens
    if max_tool_calls is not None:
        parameters["max_tool_calls"] = max_tool_calls

    payload: dict[str, Any] = {
        "model": outer_model,
        "messages": [{"role": "user", "content": question}],
        "tools": [{"type": "openrouter:fusion", "parameters": parameters}],
    }
    if force:
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


def _find_midstream_error(choice: dict[str, Any]) -> tuple[int, str] | None:
    """Detect a mid-stream error embedded in a choice (HTTP 200, finish_reason error)."""
    if choice.get("finish_reason") != "error":
        return None
    error = choice.get("error")
    if isinstance(error, dict):
        code = int(error.get("code", 0) or 0)
        message = str(error.get("message", ""))
        if code or message:
            return code, message
    return None


def parse_completion(
    data: dict[str, Any],
    *,
    panel: tuple[str, ...],
    config: ModelConfig,
    request_count: int,
) -> FusionResult:
    """Turn an OpenRouter chat-completion body into a ``FusionResult``."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    final_answer = message.get("content")
    outer_model = str(data.get("model") or config.primary_outer)

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    cost = usage.get("cost") if isinstance(usage, dict) else None
    cost_usd = float(cost) if isinstance(cost, int | float) else None

    analysis = _find_analysis(data, message)
    panel_responses = _find_panel_responses(data, message)
    failure_reason = _find_failure_reason(data, message)

    # Mid-stream error: HTTP was 200 but the choice carries an error.
    if not failure_reason and isinstance(choice, dict):
        midstream = _find_midstream_error(choice)
        if midstream is not None:
            failure_reason = midstream[1] or f"HTTP {midstream[0]}"

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
        judge=config.primary_judge,
        request_count=request_count,
        final_answer=str(final_answer) if final_answer else None,
        analysis=analysis,
        panel_responses=panel_responses,
        cost_usd=cost_usd,
        failure_reason=failure_reason,
        raw=data,
    )


def _is_rotatable(exc: FusionAPIError) -> bool:
    """Whether this error indicates the model is temporarily unavailable."""
    if exc.error_type and exc.error_type in ROTATE_ERROR_TYPES:
        return True
    return exc.status_code in ROTATE_STATUS


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Extract the Retry-After header value in seconds, if present."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


async def probe_model(client: httpx.AsyncClient, model: str) -> bool:
    """Cheap 1-token completion to check if *model* accepts requests now."""
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
            },
            timeout=PROBE_TIMEOUT,
        )
        return response.status_code < 400
    except httpx.HTTPError:
        return False


async def _post_with_retry(
    client: httpx.AsyncClient, payload: dict[str, Any], *, max_retries: int = MAX_RETRIES
) -> dict[str, Any]:
    """POST the payload, retrying transient errors. Raises on terminal failure."""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("network error on fusion attempt %d: %s", attempt, exc)
            await _backoff(attempt)
            continue

        if response.status_code in RETRY_STATUS and attempt < max_retries:
            retry_after = _parse_retry_after(response)
            logger.warning(
                "retryable HTTP %d on fusion attempt %d (retry_after=%s)",
                response.status_code,
                attempt,
                retry_after,
            )
            await _backoff(attempt, retry_after=retry_after)
            continue

        if response.status_code >= 400:
            error_type = _extract_error_type(response)
            raise FusionAPIError(
                f"OpenRouter returned HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text,
                error_type=error_type,
            )

        try:
            return response.json()  # type: ignore[no-any-return]
        except ValueError as exc:
            if attempt < max_retries:
                logger.warning("invalid JSON on attempt %d: %s", attempt, exc)
                await _backoff(attempt)
                continue
            raise FusionAPIError(
                f"OpenRouter returned invalid JSON: {exc}",
                status_code=502,
                body=response.text[:500],
                error_type="provider_unavailable",
            ) from exc

    raise FusionAPIError(
        f"fusion request failed after {max_retries + 1} attempts: {last_error}",
        status_code=None,
    )


def _extract_error_type(response: httpx.Response) -> str | None:
    """Parse ``error.metadata.error_type`` from an error response body."""
    try:
        data = response.json()
    except Exception:
        return None
    error = data.get("error")
    if isinstance(error, dict):
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            et = metadata.get("error_type")
            if isinstance(et, str):
                return et
    return None


async def _backoff(attempt: int, *, retry_after: float | None = None) -> None:
    if retry_after is not None and retry_after <= RETRY_AFTER_SHORT:
        await asyncio.sleep(retry_after)
        return
    delay = min(BACKOFF_BASE**attempt, BACKOFF_MAX)
    await asyncio.sleep(delay)


async def run_fusion(
    client: httpx.AsyncClient,
    question: str,
    config: ModelConfig,
    *,
    force: bool = True,
    panel_size: int | None = None,
    tracker: BudgetTracker | None = None,
    max_completion_tokens: int | None = None,
    max_tool_calls: int | None = None,
) -> FusionResult:
    """Execute a single fusion deliberation with automatic model rotation."""
    panels = _resolve_panels(config, panel_size)
    request_count = estimate_request_count(panels[0])

    if tracker is not None:
        await tracker.throttle_rpm(request_count)
        if not tracker.can_run(request_count):
            raise FusionBudgetError(
                f"daily free-model budget exhausted: {tracker.used_today}/{tracker.rpd_cap} "
                f"requests used, need {request_count} more."
            )

    tried: list[str] = []
    outer_models = list(config.outer)
    judge_models = list(config.judge)

    for o_idx, outer_model in enumerate(outer_models):
        # Lazy probe: primary (idx 0) is tested by the real POST; backups are probed.
        if o_idx > 0:
            if not await probe_model(client, outer_model):
                tried.append(f"{outer_model}: unavailable (probe)")
                if tracker is not None:
                    tracker.record(1)
                continue
            if tracker is not None:
                tracker.record(1)

        for p_idx, panel_set in enumerate(panels):
            for j_idx, judge_model in enumerate(judge_models):
                is_last = (
                    o_idx + 1 == len(outer_models)
                    and p_idx + 1 == len(panels)
                    and j_idx + 1 == len(judge_models)
                )
                payload, resolved_panel = build_payload(
                    question,
                    config,
                    force=force,
                    panel=panel_set,
                    outer=outer_model,
                    judge=judge_model,
                    max_completion_tokens=max_completion_tokens,
                    max_tool_calls=max_tool_calls,
                )

                try:
                    data = await _post_with_retry(
                        client, payload, max_retries=MAX_RETRIES if is_last else 1
                    )
                except FusionAPIError as exc:
                    if not is_last and _is_rotatable(exc):
                        tried.append(f"{outer_model}: HTTP {exc.status_code} ({exc.error_type})")
                        break  # outer unavailable → next outer
                    raise

                result = parse_completion(
                    data, panel=resolved_panel, config=config, request_count=request_count
                )
                result.outer = outer_model
                result.judge = judge_model

                if result.status == "error" and result.failure_reason and not is_last:
                    tried.append(f"fusion[{outer_model}/{judge_model}]: {result.failure_reason}")
                    continue  # server-side panel/judge failure → next combo

                if tracker is not None:
                    tracker.record(request_count)
                if result.cost_usd is None:
                    result.cost_usd = FREE_COST_USD
                if tried:
                    result.models_tried = tried
                return result
            else:
                continue  # judge loop exhausted without break → next panel
            break  # outer broke → next outer

    raise FusionAPIError(
        f"all model candidates exhausted; tried: {'; '.join(tried) or 'none'}",
        status_code=429,
    )


def _resolve_panels(config: ModelConfig, panel_size: int | None) -> list[tuple[str, ...]]:
    """Return panel compositions trimmed to *panel_size*, primary first."""
    from .presets import panel_candidates

    return panel_candidates(config, panel_size)


__all__ = [
    "FusionResult",
    "KeyInfo",
    "build_payload",
    "estimate_request_count",
    "get_key_info",
    "parse_completion",
    "probe_model",
    "run_fusion",
]
