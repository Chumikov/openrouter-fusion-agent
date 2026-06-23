"""Plain-text report rendering for the CLI."""

from __future__ import annotations

from typing import Any

from .fusion import FusionResult

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def render_result(result: FusionResult) -> str:
    """Render a ``FusionResult`` as a readable multi-section report."""
    lines: list[str] = []

    status_label = {
        "ok": _color("ok", GREEN),
        "degraded": _color("degraded", YELLOW),
        "error": _color("error", RED),
    }.get(result.status, result.status)
    lines.append(_color("== Fusion report ==", BOLD))
    lines.append(f"status:    {status_label}")
    lines.append(f"outer:     {result.outer}")
    lines.append(f"panel:     {', '.join(result.panel)}")
    lines.append(f"judge:     {result.judge}")
    lines.append(f"requests:  {result.request_count}  (panel + judge + outer)")
    cost = "n/a" if result.cost_usd is None else f"${result.cost_usd:.6f}"
    lines.append(f"cost:      {cost}")

    if result.models_tried:
        lines.append("")
        lines.append(_color("== Models rotated ==", YELLOW))
        for entry in result.models_tried:
            lines.append(f"  {entry}")
        lines.append(f"  {_color(f'→ using {result.outer}', GREEN)}")

    if result.failure_reason:
        lines.append("")
        lines.append(_color("== Failure ==", RED))
        lines.append(result.failure_reason)

    if result.analysis:
        lines.append("")
        lines.append(_color("== Analysis ==", BOLD))
        lines.extend(_render_analysis(result.analysis))

    if result.panel_responses:
        lines.append("")
        lines.append(_color("== Panel responses ==", BOLD))
        for item in result.panel_responses:
            if isinstance(item, dict):
                model = item.get("model", "?")
                content = item.get("content", "")
            else:
                model, content = "?", str(item)
            preview = str(content).strip().replace("\n", " ")
            if len(preview) > 300:
                preview = preview[:297] + "..."
            lines.append(f"- {_color(str(model), DIM)}: {preview}")

    if result.final_answer:
        lines.append("")
        lines.append(_color("== Final answer ==", BOLD))
        lines.append(result.final_answer)
    elif result.status != "error":
        lines.append("")
        lines.append(_color("(no final answer returned)", DIM))

    return "\n".join(lines)


def _render_analysis(analysis: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    def _block(title: str, key: str) -> None:
        value = analysis.get(key)
        if not value:
            return
        lines.append(_color(f"[{title}]", DIM))
        if isinstance(value, list):
            for item in value:
                lines.append(f"  - {_stringify(item)}")
        else:
            lines.append(f"  {_stringify(value)}")

    _block("Consensus", "consensus")
    _block("Contradictions", "contradictions")
    _block("Partial coverage", "partial_coverage")
    _block("Unique insights", "unique_insights")
    _block("Blind spots", "blind_spots")
    return lines


def _stringify(item: Any) -> str:
    if isinstance(item, dict):
        if "point" in item:
            return str(item["point"])
        if "insight" in item:
            return f"{item.get('model', '?')}: {item['insight']}"
        if "topic" in item:
            stances = item.get("stances", [])
            parts = [
                f"{s.get('model', '?')}: {s.get('stance', '')}"
                for s in stances
                if isinstance(s, dict)
            ]
            return f"{item.get('topic', '?')} -> {' | '.join(parts)}"
    return str(item)


def render_status(snapshot: dict[str, Any], key_label: str, balance: float | None) -> str:
    lines = [
        _color("== Fusion budget status ==", BOLD),
        f"key:              {key_label or '(unlabeled)'}",
        f"daily rpd cap:    {snapshot['rpd_cap']}",
        f"used today:       {snapshot['used_today']}",
        f"remaining today:  {snapshot['remaining_requests']} requests ({snapshot['runs_left']} runs)",
        f"rpm cap:          {snapshot['rpm_cap']}",
        f"balance:          {'n/a' if balance is None else f'${balance:.2f}'}",
    ]
    if balance is not None and balance < 0:
        lines.append(_color("WARNING: negative balance -> HTTP 402 even on free models.", RED))
    return "\n".join(lines)
