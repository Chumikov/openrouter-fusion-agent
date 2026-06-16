"""OpenRouter HTTP client factory and shared constants."""

from __future__ import annotations

import os

import httpx

from .errors import FusionConfigError

API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 120.0


def get_api_key() -> str:
    """Return the OpenRouter API key from the environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise FusionConfigError(
            "OPENROUTER_API_KEY is not set. Export it or pass it via the "
            "opencode mcp `environment` block."
        )
    return key


def build_client(*, timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` authenticated against OpenRouter."""
    return httpx.AsyncClient(
        base_url=API_BASE_URL,
        headers={
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
