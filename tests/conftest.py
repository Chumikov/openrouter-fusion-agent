"""Shared test fixtures."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Ensure an API key is present for client construction in tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return "test-key"


def sample_completion(
    *, answer: str = "final answer", analysis: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A representative OpenRouter chat-completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    if analysis is not None:
        message["analysis"] = analysis
    return {
        "id": "gen-test",
        "model": "qwen/qwen3-next-80b-a3b-instruct:free",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0},
    }


def sample_key_info(
    *, is_free_tier: bool = False, limit_remaining: float | None = 5.0
) -> dict[str, Any]:
    return {
        "data": {
            "label": "test-key",
            "limit": None,
            "limit_remaining": limit_remaining,
            "usage_daily": 0.0,
            "is_free_tier": is_free_tier,
        }
    }
