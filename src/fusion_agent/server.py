"""MCP server exposing fusion tools to opencode (and any MCP client).

Run with ``fusion-agent --mcp`` (stdio transport). Tools:

* ``fusion_query``  - run a multi-model deliberation on free models.
* ``fusion_status`` - show the current free-tier budget snapshot.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .budget import BudgetTracker, get_key_info
from .errors import FusionError
from .fusion import FusionResult, estimate_request_count, run_fusion
from .http import build_client
from .presets import Preset, get_preset, pick_panel

mcp = FastMCP("openrouter-fusion-agent")

_state: dict[str, Any] = {"client": None, "tracker": None, "key_info": None}


async def _ensure() -> tuple[Any, BudgetTracker, Any]:
    """Lazily build the HTTP client and budget tracker from the live key info."""
    if _state["client"] is None:
        client = build_client()
        try:
            info = await get_key_info(client)
        except Exception:
            await client.aclose()
            raise
        _state["client"] = client
        _state["key_info"] = info
        _state["tracker"] = BudgetTracker(rpd_cap=info.daily_free_rpd)
    return _state["client"], _state["tracker"], _state["key_info"]


@mcp.tool()
async def fusion_query(
    question: str,
    force: bool = True,
    panel_size: int | None = None,
    preset: str = "quality",
) -> dict[str, Any]:
    """Run an OpenRouter Fusion multi-model deliberation using free models.

    Use for research, compare/contrast, expert critique, or any task where the
    cost of being wrong outweighs a few extra completions. Returns the outer
    model's final answer plus best-effort structured analysis and panel
    responses when OpenRouter echoes them.

    Args:
        question: The prompt to deliberate on.
        force: When True (default) the outer model is required to invoke fusion.
        panel_size: Optional panel size (1-3). Smaller when budget is low.
        preset: "quality" (default) or "budget".
    """
    try:
        client, tracker, _info = await _ensure()
        resolved: Preset = get_preset(preset)
        result: FusionResult = await run_fusion(
            client,
            question,
            resolved,
            force=force,
            panel_size=panel_size,
            tracker=tracker,
        )
        return result.to_dict()
    except FusionError as exc:
        return {"status": "error", "failure_reason": exc.__class__.__name__, "error": str(exc)}


@mcp.tool()
async def fusion_status() -> dict[str, Any]:
    """Return the OpenRouter free-tier budget snapshot for the current session."""
    try:
        _client, tracker, info = await _ensure()
        panel = pick_panel(get_preset("quality"))
        per_run = estimate_request_count(panel)
        return {
            "key_label": info.label,
            "is_free_tier": info.is_free_tier,
            "daily_free_rpd": info.daily_free_rpd,
            "limit": info.limit,
            "limit_remaining": info.limit_remaining,
            "has_negative_balance": info.has_negative_balance,
            "budget": tracker.snapshot(per_run),
        }
    except FusionError as exc:
        return {"status": "error", "failure_reason": exc.__class__.__name__, "error": str(exc)}


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
