from __future__ import annotations

import pytest

from fusion_agent.budget import (
    FREE_RPD_NO_CREDITS,
    FREE_RPD_WITH_CREDITS,
    BudgetTracker,
    KeyInfo,
)


def make_info(*, is_free_tier: bool = False, limit_remaining: float | None = 5.0) -> KeyInfo:
    return KeyInfo(
        label="k",
        is_free_tier=is_free_tier,
        limit=None,
        limit_remaining=limit_remaining,
        usage_daily=0.0,
    )


def test_daily_free_rpd_depends_on_tier() -> None:
    assert make_info(is_free_tier=True).daily_free_rpd == FREE_RPD_NO_CREDITS
    assert make_info(is_free_tier=False).daily_free_rpd == FREE_RPD_WITH_CREDITS


def test_negative_balance_detection() -> None:
    assert make_info(limit_remaining=-1.0).has_negative_balance is True
    assert make_info(limit_remaining=0.0).has_negative_balance is False
    assert make_info(limit_remaining=None).has_negative_balance is False


def test_can_run_and_record() -> None:
    tracker = BudgetTracker(rpd_cap=10)
    assert tracker.can_run(5) is True
    assert tracker.requests_left(5) == 2
    tracker.record(5)
    assert tracker.used_today == 5
    assert tracker.remaining_requests == 5
    assert tracker.can_run(6) is False
    tracker.record(5)
    assert tracker.requests_left(5) == 0


def test_requests_left_handles_zero_per_run() -> None:
    tracker = BudgetTracker(rpd_cap=10)
    assert tracker.requests_left(0) == 0


@pytest.mark.asyncio
async def test_throttle_rpm_records_slots() -> None:
    tracker = BudgetTracker(rpd_cap=1000)
    await tracker.throttle_rpm(5)
    assert len(tracker._rpm_window) == 5


@pytest.mark.asyncio
async def test_throttle_rpm_skips_when_n_exceeds_cap() -> None:
    tracker = BudgetTracker(rpd_cap=1000)
    # Should not loop forever when a single run exceeds the RPM cap.
    await tracker.throttle_rpm(50)
    assert tracker._rpm_window == []


def test_snapshot_shape() -> None:
    tracker = BudgetTracker(rpd_cap=1000)
    tracker.record(5)
    snap = tracker.snapshot(5)
    assert snap["used_today"] == 5
    assert snap["remaining_requests"] == 995
    assert snap["runs_left"] == 199
    assert snap["rpd_cap"] == 1000
