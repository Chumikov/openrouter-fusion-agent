"""Free-model presets for the panel and judge.

The panel is deliberately family-diverse so the models produce less correlated
answers (OpenAI + NVIDIA + Meta for Quality; Google + NVIDIA + OpenAI for
Budget). Every model here advertises native tool support, which is required
because the fusion pipeline enables ``openrouter:web_search`` /
``openrouter:web_fetch`` on every panel and judge call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """A resolved fusion configuration."""

    name: str
    outer: str
    panel: tuple[str, ...]
    judge: str


# Quality preset: strongest diverse free models.
QUALITY = Preset(
    name="quality",
    outer="qwen/qwen3-next-80b-a3b-instruct:free",
    panel=(
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    judge="nvidia/nemotron-3-ultra-550b-a55b:free",
)

# Budget preset: smaller / faster free models.
BUDGET = Preset(
    name="budget",
    outer="qwen/qwen3-next-80b-a3b-instruct:free",
    panel=(
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-20b:free",
    ),
    judge="nvidia/nemotron-3-super-120b-a12b:free",
)

_PRESETS: dict[str, Preset] = {"quality": QUALITY, "budget": BUDGET}


def get_preset(name: str) -> Preset:
    """Return the named preset (``quality`` or ``budget``)."""
    key = name.strip().lower()
    if key not in _PRESETS:
        available = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {available}")
    return _PRESETS[key]


def pick_panel(preset: Preset, size: int | None = None) -> tuple[str, ...]:
    """Return ``size`` panel models (defaults to the full panel).

    Keeps the most diverse models first, so shrinking for budget reasons still
    leaves a healthy mix of families.
    """
    if size is None:
        return preset.panel
    if not 1 <= size <= len(preset.panel):
        raise ValueError(f"panel_size must be between 1 and {len(preset.panel)}, got {size}")
    return preset.panel[:size]
