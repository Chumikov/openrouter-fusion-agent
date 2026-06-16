# openrouter-fusion-agent

Multi-model deliberation agent for [OpenRouter Fusion](https://openrouter.ai/docs/guides/features/server-tools/fusion), running **entirely on free models**. Provides a CLI and an [MCP](https://modelcontextprotocol.io) server that you can plug into [opencode](https://opencode.ai) (or any MCP client).

Fusion runs a **panel** of models in parallel (each with web search), a **judge** compares their answers and returns structured analysis — consensus, contradictions, unique insights, blind spots — and your outer model writes a stronger final answer. OpenRouter ships this with expensive paid defaults; this project reconfigures it with free (`:free`) models and adds **budget-aware safeguards** against OpenRouter's free-tier rate limits.

## Features

- 100% free models (OpenRouter `:free` variants) — $0 per run.
- Two presets: **quality** (diverse strong panel) and **budget** (smaller / faster).
- Budget-aware: reads `GET /api/v1/key`, blocks on negative balance (HTTP 402), tracks the 50/1000 daily free-request cap, throttles to 20 RPM, retries on `429`.
- **MCP server** (stdio) with `fusion_query` and `fusion_status` tools — native for opencode.
- **CLI** one-shot + interactive REPL.
- Typed (mypy strict), tested (pytest), MIT-licensed.

## Free model presets

The panel is deliberately family-diverse so models produce less correlated answers.

| Preset | Outer (decides + writes) | Panel (parallel) | Judge (analysis) |
| --- | --- | --- | --- |
| `quality` (default) | `qwen/qwen3-next-80b-a3b-instruct:free` | `openai/gpt-oss-120b:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `meta-llama/llama-3.3-70b-instruct:free` | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `budget` | `qwen/qwen3-next-80b-a3b-instruct:free` | `google/gemma-4-26b-a4b-it:free`, `nvidia/nemotron-3-nano-30b-a3b:free`, `openai/gpt-oss-20b:free` | `nvidia/nemotron-3-super-120b-a12b:free` |

A run costs roughly `len(panel) + 2` completions (5 for the default panel).

## Free-tier limits (why budget awareness matters)

OpenRouter free models are limited to **20 requests/minute** and a **daily cap** of **50 requests/day** if you've purchased **< $10** of credits, or **1000/day** at **≥ $10** (the calls themselves still cost $0). A negative account balance raises HTTP **402 even on free models**. This agent reads `GET /api/v1/key` to detect `is_free_tier` and the balance, and enforces the caps so you don't hit hard failures mid-run. Tip: adding ≥ $10 of credits lifts the daily cap to 1000 while keeping every call free.

## Installation

Requires Python ≥ 3.11.

```bash
# from source
git clone https://github.com/Chumikov/openrouter-fusion-agent
cd openrouter-fusion-agent
uv sync                          # or: pip install -e ".[dev]"

# or once published to PyPI
uvx openrouter-fusion-agent --help
```

Set your API key:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

## CLI usage

```bash
# one-shot deliberation
fusion-agent "Compare ridge, lasso and elastic-net regression. Where does each shine?"

# REPL
fusion-agent
fusion> /status
fusion> Compare the strongest arguments for and against a carbon tax.
fusion> /quit
```

REPL commands: `/status`, `/force on|off`, `/panel 1|2|3`, `/preset quality|budget`, `/budget <n>` (override daily cap), `/help`, `/quit`.

Flags: `--force on|off` (default `on` — guarantees fusion is invoked), `--panel N`, `--preset quality|budget`, `--budget N`.

## opencode integration (MCP)

Add the agent as a local MCP server in your `opencode.json` (see [`examples/opencode.json`](examples/opencode.json)):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "fusion": {
      "type": "local",
      "command": ["uvx", "openrouter-fusion-agent", "--mcp"],
      "enabled": true,
      "environment": { "OPENROUTER_API_KEY": "{env:OPENROUTER_API_KEY}" },
      "timeout": 10000
    }
  },
  "experimental": { "mcp_timeout": 90000 }
}
```

Notes:
- `experimental.mcp_timeout` is raised to ~90s because a fusion run takes 30–60s.
- Until the package is on PyPI, point `command` at a local checkout instead:
  `["uv", "run", "--directory", "/path/to/openrouter-fusion-agent", "fusion-agent", "--mcp"]`
- Restart opencode after editing config (it loads config once at startup).

Optionally drop the example skill from [`examples/skill/fusion/SKILL.md`](examples/skill/fusion/SKILL.md) into `.opencode/skills/fusion/SKILL.md` so opencode knows **when** to call fusion automatically.

Then in a session:

```
use the fusion tool to survey the arguments for and against universal basic income
```

## How it works

```
your question ─▶ outer model ─calls openrouter:fusion─▶ panel (≤3 free models + web_search)
                                                            │
                                                            ▼
                                          judge (free model + web_search) ─structured analysis─▶ outer model ─▶ final answer
```

The outer model invokes OpenRouter's `openrouter:fusion` server tool. The judge returns a structured analysis (consensus / contradictions / partial coverage / unique insights / blind spots) which the outer model consumes to write the final answer. The structured analysis is consumed internally; this agent surfaces it best-effort when OpenRouter echoes it, and always returns the final answer, cost, and which models ran.

## Development

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy src/fusion_agent
uv run pytest -q
```

CI runs the same on every push/PR (`.github/workflows/ci.yml`).

## License

MIT © Chumikov
