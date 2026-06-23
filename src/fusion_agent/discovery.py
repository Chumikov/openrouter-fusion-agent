"""Runtime discovery and selection of free models from OpenRouter.

Provides ``fusion_refresh_models`` (MCP) and ``refresh-models`` (CLI) with the
same pipeline: query ``GET /api/v1/models``, filter for free models with tool
support, rank by parameter count, assign to roles (outer / panel / judge) with
family diversity, and persist the result to a JSON file that ``presets.py``
reads at startup.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import httpx

logger = logging.getLogger("fusion_agent")

MODEL_CATALOG_CACHE_TTL = 3600  # seconds

_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b", re.IGNORECASE)


@dataclass(frozen=True)
class ModelInfo:
    """A free OpenRouter model with the metadata we select on."""

    id: str
    family: str
    param_count: float | None
    context_length: int
    name: str


class ModelSelection(TypedDict):
    """Result of :func:`select_models` — ordered candidate lists per role."""

    outer: list[str]
    panel: list[list[str]]
    judge: list[str]


def parse_param_count(*texts: str) -> float | None:
    """Extract the largest ``NNb`` parameter count (in billions) from *texts*.

    ``"nemotron-3-ultra-550b-a55b"`` → 550 (not 55); ``"lfm-2.5-1.2b"`` → 1.2;
    ``"north-mini-code"`` → None.
    """
    matches: list[float] = []
    for text in texts:
        matches.extend(float(m) for m in _PARAM_RE.findall(text))
    return max(matches) if matches else None


def _family(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


async def fetch_free_tool_models(client: httpx.AsyncClient) -> list[ModelInfo]:
    """Return free models with tool support from ``GET /api/v1/models``."""
    response = await client.get("/models")
    response.raise_for_status()
    raw_models = response.json().get("data", [])

    result: list[ModelInfo] = []
    for entry in raw_models:
        model_id = str(entry.get("id", ""))
        if not model_id.endswith(":free"):
            continue
        supported = entry.get("supported_parameters", [])
        if "tools" not in supported:
            continue
        name = str(entry.get("name", ""))
        result.append(
            ModelInfo(
                id=model_id,
                family=_family(model_id),
                param_count=parse_param_count(model_id, name),
                context_length=int(entry.get("context_length", 0) or 0),
                name=name,
            )
        )
    return result


def pick_diverse(sorted_pool: list[ModelInfo], count: int) -> list[ModelInfo]:
    """Pick *count* models preferring distinct families.

    *sorted_pool* must be pre-sorted by preference (strongest first).  We take
    one model per family in the first pass, then fill from the remainder.
    """
    picked: list[ModelInfo] = []
    seen_families: set[str] = set()

    for model in sorted_pool:
        if model.family not in seen_families:
            picked.append(model)
            seen_families.add(model.family)
        if len(picked) >= count:
            return picked

    # Not enough distinct families — fill with remaining models.
    for model in sorted_pool:
        if model not in picked:
            picked.append(model)
        if len(picked) >= count:
            break

    return picked


def select_models(models: list[ModelInfo], *, min_b: float = 20) -> ModelSelection:
    """Select ordered candidate lists for outer / panel / judge roles.

    Returns a :class:`ModelSelection` ordered by priority (primary first, then
    backups).
    """
    pool = [m for m in models if m.param_count is not None and m.param_count >= min_b]
    if len(pool) < 5:
        # Too few large models — relax the threshold and take the top by size.
        pool = sorted(models, key=lambda m: m.param_count or 0, reverse=True)[:8]

    sorted_pool = sorted(pool, key=lambda m: m.param_count or 0, reverse=True)

    outer = pick_diverse(sorted_pool, 3)
    judge_pool = [m for m in sorted_pool if m.family != outer[0].family] if outer else sorted_pool
    if not judge_pool:
        judge_pool = sorted_pool
    judge = pick_diverse(judge_pool, 2)

    # Primary panel: 3 models from 3 distinct families.
    panel_primary = pick_diverse(sorted_pool, 3)
    # Backup panel: alternative models, prioritising families not in primary.
    used_ids = {m.id for m in panel_primary}
    backup_pool = [m for m in sorted_pool if m.id not in used_ids]
    backup_panel = pick_diverse(backup_pool, 3) if backup_pool else []
    # If not enough distinct families for a full backup, reuse primary models
    # to ensure at least 2 in the backup composition.
    if len(backup_panel) < 2:
        for m in panel_primary:
            if m not in backup_panel:
                backup_panel.append(m)
            if len(backup_panel) >= 2:
                break

    panels = [panel_primary]
    if len(backup_panel) >= 2:
        panels.append(backup_panel)

    return ModelSelection(
        outer=[m.id for m in outer],
        panel=[[m.id for m in p] for p in panels],
        judge=[m.id for m in judge],
    )


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


def models_file_path() -> Path:
    """Resolve the models JSON path (XDG-aware, env-overridable)."""
    env = os.environ.get("FUSION_MODELS_FILE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "openrouter-fusion-agent" / "models.json"
    return Path.home() / ".config" / "openrouter-fusion-agent" / "models.json"


def save_selection(path: Path, selection: ModelSelection, *, total_found: int) -> Path:
    """Write the selection dict to *path* with metadata. Returns *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "version": 1,
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "openrouter:/api/v1/models",
        "total_free_tool_models": total_found,
        **selection,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_selection(path: Path) -> dict[str, object] | None:
    """Read and validate the selection file. Returns None if missing/invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("outer", "panel", "judge"):
        if key not in data:
            return None
    return data


__all__ = [
    "MODEL_CATALOG_CACHE_TTL",
    "ModelInfo",
    "ModelSelection",
    "fetch_free_tool_models",
    "load_selection",
    "models_file_path",
    "parse_param_count",
    "pick_diverse",
    "save_selection",
    "select_models",
]
