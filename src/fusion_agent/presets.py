"""Model configuration: ordered candidate lists for each fusion role.

The configuration is loaded from ``models.json`` (written by
``refresh-models``) when available, falling back to ``DEFAULT_CONFIG``.
Candidates are ordered by priority — the first entry is the primary model,
the rest are backups tried in order when the primary is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .discovery import load_selection, models_file_path

logger = logging.getLogger("fusion_agent")


@dataclass(frozen=True)
class ModelConfig:
    """Ordered candidate models for each fusion role (primary first)."""

    outer: tuple[str, ...]
    panel: tuple[tuple[str, ...], ...]
    judge: tuple[str, ...]

    @property
    def primary_outer(self) -> str:
        return self.outer[0]

    @property
    def primary_judge(self) -> str:
        return self.judge[0]

    @property
    def primary_panel(self) -> tuple[str, ...]:
        return self.panel[0]


DEFAULT_CONFIG = ModelConfig(
    outer=(
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    panel=(
        (
            "openai/gpt-oss-120b:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ),
    ),
    judge=(
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openai/gpt-oss-120b:free",
    ),
)


def _from_dict(data: dict[str, object]) -> ModelConfig:
    """Build a ``ModelConfig`` from a parsed selection file."""
    outer_raw = data.get("outer")
    panel_raw = data.get("panel")
    judge_raw = data.get("judge")
    if not isinstance(outer_raw, list) or not isinstance(panel_raw, list):
        raise ValueError("invalid model list format in selection file")
    if not isinstance(judge_raw, list):
        raise ValueError("invalid model list format in selection file")

    outer = tuple(str(m) for m in outer_raw)
    panel = tuple(tuple(str(m) for m in p) for p in panel_raw if isinstance(p, list))
    judge = tuple(str(m) for m in judge_raw)
    if not outer or not panel or not judge:
        raise ValueError("empty model list in selection file")
    return ModelConfig(outer=outer, panel=panel, judge=judge)


_cached: ModelConfig | None = None


def get_config(path: Path | None = None) -> ModelConfig:
    """Return the active model configuration (cached, lazy-loaded).

    Reads from *path* (defaults to :func:`~discovery.models_file_path`) on the
    first call; subsequent calls return the cache.  Use :func:`reload_config`
    to invalidate.
    """
    global _cached
    if _cached is not None:
        return _cached
    file_path = path or models_file_path()
    data = load_selection(file_path)
    if data is not None:
        try:
            _cached = _from_dict(data)
            logger.info("loaded model config from %s", file_path)
            return _cached
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("invalid model config in %s (%s); using defaults", file_path, exc)
    _cached = DEFAULT_CONFIG
    return _cached


def reload_config() -> None:
    """Invalidate the cache so the next :func:`get_config` re-reads the file."""
    global _cached
    _cached = None


MIN_PANEL_SIZE = 2


def pick_panel(config: ModelConfig, size: int | None = None) -> tuple[str, ...]:
    """Return the primary panel, optionally trimmed to *size* members.

    The minimum panel size is :data:`MIN_PANEL_SIZE` (2).  Passing a smaller
    value raises ``ValueError``.
    """
    primary = config.primary_panel
    if size is None:
        return primary
    if size < MIN_PANEL_SIZE:
        raise ValueError(f"panel_size must be >= {MIN_PANEL_SIZE}, got {size}")
    if size > len(primary):
        raise ValueError(f"panel_size must be <= {len(primary)}, got {size}")
    return primary[:size]


def panel_candidates(config: ModelConfig, size: int | None = None) -> list[tuple[str, ...]]:
    """Return all panel compositions (trimmed to *size*), primary first.

    Compositions smaller than :data:`MIN_PANEL_SIZE` are skipped.
    """
    result: list[tuple[str, ...]] = []
    for panel in config.panel:
        trimmed = panel[:size] if size is not None else panel
        if len(trimmed) >= MIN_PANEL_SIZE:
            result.append(trimmed)
    return result


__all__ = [
    "DEFAULT_CONFIG",
    "MIN_PANEL_SIZE",
    "ModelConfig",
    "get_config",
    "panel_candidates",
    "pick_panel",
    "reload_config",
]
