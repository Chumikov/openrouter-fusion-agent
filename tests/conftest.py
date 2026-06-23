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
    *,
    answer: str = "final answer",
    analysis: dict[str, Any] | None = None,
    model: str = "qwen/qwen3-next-80b-a3b-instruct:free",
) -> dict[str, Any]:
    """A representative OpenRouter chat-completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": answer}
    if analysis is not None:
        message["analysis"] = analysis
    return {
        "id": "gen-test",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0},
    }


def sample_completion_midstream_error(
    *,
    code: int = 503,
    message: str = "Provider overloaded",
    error_type: str = "provider_overloaded",
) -> dict[str, Any]:
    """A chat-completion body carrying a mid-stream error (HTTP 200, finish_reason error)."""
    return {
        "id": "gen-test",
        "model": "qwen/qwen3-next-80b-a3b-instruct:free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "error",
                "error": {
                    "code": code,
                    "message": message,
                    "metadata": {"error_type": error_type},
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "cost": 0.0},
    }


def sample_error_response(
    *,
    code: int = 429,
    error_type: str = "rate_limit_exceeded",
    message: str = "Rate limit exceeded",
) -> dict[str, Any]:
    """An OpenRouter error response body."""
    return {
        "error": {
            "code": code,
            "message": message,
            "metadata": {"error_type": error_type},
        }
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


def sample_models_catalog() -> dict[str, Any]:
    """A mock ``GET /api/v1/models`` response with a diverse set of free models."""
    models = [
        {
            "id": "qwen/qwen3-next-80b-a3b-instruct:free",
            "name": "Qwen: Qwen3 Next 80B A3B Instruct (free)",
            "context_length": 262144,
            "supported_parameters": ["tools", "temperature"],
        },
        {
            "id": "openai/gpt-oss-120b:free",
            "name": "OpenAI: gpt-oss-120b (free)",
            "context_length": 131072,
            "supported_parameters": ["tools"],
        },
        {
            "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "name": "NVIDIA: Nemotron 3 Ultra 550B A55B (free)",
            "context_length": 1000000,
            "supported_parameters": ["tools"],
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Meta: Llama 3.3 70B Instruct (free)",
            "context_length": 131072,
            "supported_parameters": ["tools"],
        },
        {
            "id": "google/gemma-4-26b-a4b-it:free",
            "name": "Google: Gemma 4 26B A4B (free)",
            "context_length": 262144,
            "supported_parameters": ["tools"],
        },
        {
            "id": "openai/gpt-oss-20b:free",
            "name": "OpenAI: gpt-oss-20b (free)",
            "context_length": 131072,
            "supported_parameters": ["tools"],
        },
        {
            "id": "liquid/lfm-2.5-1.2b-thinking:free",
            "name": "LiquidAI: LFM2.5-1.2B-Thinking (free)",
            "context_length": 32768,
            "supported_parameters": ["tools"],
        },
        {
            "id": "cohere/north-mini-code:free",
            "name": "Cohere: North Mini Code (free)",
            "context_length": 256000,
            # No tools — should be filtered out.
            "supported_parameters": ["temperature"],
        },
    ]
    return {"data": models}
