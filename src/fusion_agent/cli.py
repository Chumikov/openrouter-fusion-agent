"""Command-line interface and interactive REPL."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys

from . import __version__
from . import server as server_module
from .budget import BudgetTracker, get_key_info
from .errors import FusionError
from .fusion import run_fusion
from .http import build_client
from .install import install_skill, print_config
from .presets import get_preset
from .render import render_result, render_status

HELP_TEXT = """\
fusion-agent commands (inside the REPL):
  <question>        run a fusion deliberation on the given text
  /status           show the current free-tier budget snapshot
  /force on|off     toggle forcing fusion on every run (default: on)
  /panel <1|2|3>    set the panel size (default: 3)
  /preset quality|budget   switch preset (default: quality)
  /budget <n>       override the daily RPD cap manually
  /help             show this help
  /quit             exit
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusion-agent",
        description="Multi-model deliberation agent for OpenRouter Fusion (free models).",
    )
    parser.add_argument("--version", action="version", version=f"fusion-agent {__version__}")
    parser.add_argument("question", nargs="*", help="question to deliberate on; omit for REPL")
    parser.add_argument(
        "--mcp", action="store_true", help="run as an MCP stdio server (for opencode)"
    )
    parser.add_argument(
        "--force", choices=["on", "off"], default="on", help="force fusion on every run"
    )
    parser.add_argument("--panel", type=int, default=None, help="panel size (1-3)")
    parser.add_argument(
        "--preset", choices=["quality", "budget"], default="quality", help="model preset"
    )
    parser.add_argument("--budget", type=int, default=None, help="override daily RPD cap")
    return parser


async def _one_shot(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        print('error: provide a question, e.g.  fusion-agent "compare X and Y"', file=sys.stderr)
        return 2

    async with build_client() as client:
        info = await get_key_info(client)
        if info.has_negative_balance:
            print(
                f"error: OpenRouter balance is negative (${info.limit_remaining:.2f}); "
                "free models also return HTTP 402. Top up your balance.",
                file=sys.stderr,
            )
            return 3
        rpd_cap = args.budget if args.budget is not None else info.daily_free_rpd
        tracker = BudgetTracker(rpd_cap=rpd_cap)
        preset = get_preset(args.preset)
        result = await run_fusion(
            client,
            question,
            preset,
            force=(args.force == "on"),
            panel_size=args.panel,
            tracker=tracker,
        )
    print(render_result(result))
    return 0 if result.ok else 1


async def _repl() -> int:
    force = True
    panel: int | None = None
    preset_name = "quality"
    rpd_override: int | None = None

    print(HELP_TEXT)
    while True:
        try:
            line = input("fusion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if not line.startswith("/"):
            async with build_client() as client:
                info = await get_key_info(client)
                if info.has_negative_balance:
                    print(
                        f"error: negative balance (${info.limit_remaining:.2f}) -> HTTP 402 on free models.",
                        file=sys.stderr,
                    )
                    continue
                rpd_cap = rpd_override if rpd_override is not None else info.daily_free_rpd
                tracker = BudgetTracker(rpd_cap=rpd_cap)
                try:
                    result = await run_fusion(
                        client,
                        line,
                        get_preset(preset_name),
                        force=force,
                        panel_size=panel,
                        tracker=tracker,
                    )
                except FusionError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    continue
            print()
            print(render_result(result))
            continue

        parts = shlex.split(line[1:])
        cmd = parts[0] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in {"quit", "exit", "q"}:
            return 0
        if cmd == "help":
            print(HELP_TEXT)
        elif cmd == "status":
            async with build_client() as client:
                info = await get_key_info(client)
                rpd_cap = rpd_override if rpd_override is not None else info.daily_free_rpd
                tracker = BudgetTracker(rpd_cap=rpd_cap)
                from .fusion import estimate_request_count
                from .presets import pick_panel

                per_run = estimate_request_count(pick_panel(get_preset(preset_name), panel))
                print(render_status(tracker.snapshot(per_run), info.label, info.limit_remaining))
        elif cmd == "force":
            force = arg.lower() in {"on", "true", "1", "yes"}
            print(f"force = {force}")
        elif cmd == "panel":
            try:
                panel = int(arg) if arg else None
            except ValueError:
                panel = None
            print(f"panel = {panel}")
        elif cmd == "preset":
            if arg in {"quality", "budget"}:
                preset_name = arg
            print(f"preset = {preset_name}")
        elif cmd == "budget":
            try:
                rpd_override = int(arg) if arg else None
            except ValueError:
                rpd_override = None
            print(f"rpd_override = {rpd_override}")
        else:
            print(f"unknown command: /{cmd}  (try /help)")


def _run_install_skill(rest: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="fusion-agent install-skill",
        description="Install the opencode fusion skill (SKILL.md).",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--project", action="store_true", help="install into ./.opencode/skills/fusion (default)"
    )
    scope.add_argument(
        "--global",
        dest="global_",
        action="store_true",
        help="install into ~/.config/opencode/skills/fusion",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing SKILL.md")
    ns = parser.parse_args(rest)
    target_scope = "global" if ns.global_ else "project"
    try:
        target = install_skill(scope=target_scope, force=ns.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"installed skill -> {target}")
    print("restart opencode to load it.")
    return 0


def _run_print_config(_rest: list[str]) -> int:
    print(print_config())
    return 0


_SUBCOMMANDS = {
    "install-skill": _run_install_skill,
    "print-config": _run_print_config,
}


def main() -> None:
    """Console-script entry point."""
    argv = sys.argv[1:]
    if argv and argv[0] in _SUBCOMMANDS:
        rc = _SUBCOMMANDS[argv[0]](argv[1:])
        raise SystemExit(rc)

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mcp:
        server_module.main()
        return

    try:
        rc = asyncio.run(_one_shot(args) if args.question else _repl())
    except FusionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        rc = 1
    except KeyboardInterrupt:
        rc = 130
    raise SystemExit(rc)
