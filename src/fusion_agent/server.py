"""MCP server exposing fusion tools to opencode (and any MCP client).

Run with ``fusion-agent --mcp`` (stdio transport). Tools:

* ``fusion_query``         - run a multi-model deliberation on free models.
* ``fusion_status``        - show the current free-tier budget snapshot.
* ``fusion_refresh_models`` - discover current free models and update the local selection.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from .budget import BudgetTracker, get_key_info
from .discovery import (
    fetch_free_tool_models,
    models_file_path,
    save_selection,
    select_models,
)
from .errors import FusionError
from .fusion import FusionResult, estimate_request_count, run_fusion
from .http import build_client
from .presets import get_config, pick_panel, reload_config

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
    *,
    ctx: Context[Any],  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Run an OpenRouter Fusion multi-model deliberation using free models.

    Use for research, compare/contrast, expert critique, or any task where the
    cost of being wrong outweighs a few extra completions. Returns the outer
    model's final answer plus best-effort structured analysis and panel
    responses when OpenRouter echoes them. If a model is unavailable (429/5xx),
    backup models are tried automatically.

    Args:
        question: The prompt to deliberate on.
        force: When True (default) the outer model is required to invoke fusion.
        panel_size: Panel size: 2 or 3 (default: 3). Cannot be less than 2.
    """
    try:
        client, tracker, _info = await _ensure()
        config = get_config()

        async def report(msg: str) -> None:
            await ctx.info(msg)

        result: FusionResult = await run_fusion(
            client,
            question,
            config,
            force=force,
            panel_size=panel_size,
            tracker=tracker,
            on_progress=report,
        )
        return result.to_dict()
    except FusionError as exc:
        return {"status": "error", "failure_reason": exc.__class__.__name__, "error": str(exc)}


@mcp.tool()
async def fusion_status() -> dict[str, Any]:
    """Return the OpenRouter free-tier budget snapshot for the current session."""
    try:
        _client, tracker, info = await _ensure()
        config = get_config()
        panel = pick_panel(config)
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


@mcp.tool()
async def fusion_refresh_models(
    min_b: int = 20,
    *,
    ctx: Context[Any],  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Discover current free models from OpenRouter and update the local selection.

    Queries ``GET /api/v1/models``, filters for free models with tool support,
    ranks by parameter count, assigns to roles (outer / panel / judge) with
    family diversity, and persists the result. The next ``fusion_query`` call
    will use the updated models automatically.

    Args:
        min_b: Minimum parameter count (in billions) for model selection. Models
            smaller than this are excluded. Defaults to 20.
    """
    try:
        client, _tracker, _info = await _ensure()
        await ctx.info("Querying OpenRouter model catalog...")
        models = await fetch_free_tool_models(client)
        await ctx.info(f"Found {len(models)} free models with tool support")
        selection = select_models(models, min_b=float(min_b))
        path = save_selection(models_file_path(), selection, total_found=len(models))
        reload_config()
        await ctx.info(f"Saved to {path}")
        return {
            "path": str(path),
            "models_found": len(models),
            "outer": selection["outer"],
            "panel": selection["panel"],
            "judge": selection["judge"],
        }
    except FusionError as exc:
        return {"status": "error", "failure_reason": exc.__class__.__name__, "error": str(exc)}


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
