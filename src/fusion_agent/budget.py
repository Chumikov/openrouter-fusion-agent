"""Free-tier budget tracking for OpenRouter.

OpenRouter does not expose a "free requests remaining today" counter, so we
combine two sources of truth:

* ``GET /api/v1/key`` returns ``is_free_tier`` (governs the daily cap: 50
  requests/day below $10 of purchased credits, 1000/day at or above) plus the
  credit balance (a negative balance raises HTTP 402 *even on free models*).
* An in-process ``BudgetTracker`` counts the model completions we issue
  (roughly ``len(panel) + 2`` per fusion run) and throttles to the 20 RPM cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anyio
import httpx

# OpenRouter free-model limits (see https://openrouter.ai/docs/api/reference/limits).
FREE_RPM = 20
FREE_RPD_NO_CREDITS = 50
FREE_RPD_WITH_CREDITS = 1000
CREDITS_THRESHOLD_USD = 10.0
RPM_WINDOW_SECONDS = 60.0


@dataclass
class KeyInfo:
    """Snapshot of an OpenRouter API key's limits and usage."""

    label: str
    is_free_tier: bool
    limit: float | None
    limit_remaining: float | None
    usage_daily: float

    @property
    def daily_free_rpd(self) -> int:
        """Daily free-model request cap (50 or 1000)."""
        return FREE_RPD_NO_CREDITS if self.is_free_tier else FREE_RPD_WITH_CREDITS

    @property
    def has_negative_balance(self) -> bool:
        """Whether the balance is below zero (would cause HTTP 402 even on free)."""
        return self.limit_remaining is not None and self.limit_remaining < 0


async def get_key_info(client: httpx.AsyncClient) -> KeyInfo:
    """Fetch the key limits/usage snapshot from ``GET /api/v1/key``."""
    response = await client.get("/key")
    response.raise_for_status()
    data = response.json().get("data", {})

    def _as_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    return KeyInfo(
        label=str(data.get("label") or ""),
        is_free_tier=bool(data.get("is_free_tier", True)),
        limit=_as_float(data.get("limit")),
        limit_remaining=_as_float(data.get("limit_remaining")),
        usage_daily=_as_float(data.get("usage_daily")) or 0.0,
    )


@dataclass
class BudgetTracker:
    """In-process tracker for the daily and per-minute free-model limits."""

    rpd_cap: int
    used_today: int = 0
    _rpm_window: list[float] = field(default_factory=list)

    @property
    def remaining_requests(self) -> int:
        return max(0, self.rpd_cap - self.used_today)

    def requests_left(self, per_run: int) -> int:
        """How many full fusion runs of ``per_run`` requests each still fit today."""
        if per_run <= 0:
            return 0
        return self.remaining_requests // per_run

    def can_run(self, request_count: int) -> bool:
        return self.used_today + request_count <= self.rpd_cap

    def record(self, request_count: int) -> None:
        self.used_today += request_count

    async def throttle_rpm(self, n: int = 1) -> None:
        """Block until ``n`` slots are free within the 20 RPM rolling window."""
        if n >= FREE_RPM:
            # A single run cannot exceed the cap meaningfully; just proceed.
            return
        while True:
            now = time.monotonic()
            cutoff = now - RPM_WINDOW_SECONDS
            self._rpm_window = [t for t in self._rpm_window if t >= cutoff]
            if len(self._rpm_window) + n <= FREE_RPM:
                stamp = time.monotonic()
                self._rpm_window.extend([stamp] * n)
                return
            wait = self._rpm_window[0] + RPM_WINDOW_SECONDS - now
            if wait > 0:
                await anyio.sleep(min(wait, RPM_WINDOW_SECONDS))

    def snapshot(self, per_run: int) -> dict[str, object]:
        return {
            "rpd_cap": self.rpd_cap,
            "used_today": self.used_today,
            "remaining_requests": self.remaining_requests,
            "runs_left": self.requests_left(per_run),
            "rpm_cap": FREE_RPM,
        }
