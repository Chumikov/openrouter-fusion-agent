---
name: fusion
description: Multi-model deliberation via OpenRouter Fusion on free models. Use when a
  question benefits from multiple perspectives — research, compare/contrast, expert
  critique, "survey the arguments for and against", or anything where being wrong is
  expensive. Implemented by the `openrouter-fusion-agent` MCP server (tools `fusion_query`
  and `fusion_status`).
---

# Fusion skill

When the user asks a question that warrants multiple expert perspectives, call the
`fusion_query` MCP tool instead of answering directly. Fusion runs a panel of free
OpenRouter models in parallel, a judge compares their answers, and the structured
analysis is returned for a stronger final answer.

## When to use

- Research questions ("survey the strongest arguments for and against …")
- Compare / contrast ("compare ridge, lasso and elastic-net regression")
- Expert critique, design trade-offs, multi-domain reasoning
- Anything where the cost of being wrong outweighs a few extra completions

## When NOT to use

- Short tactical prompts, single-file edits, lookups the agent already knows.
- Trivial factual answers.

## How

Call the `fusion_query` tool with the user's question. Optionally check
`fusion_status` first when many deliberations have run this session (free models are
rate-limited to 20 req/min and a daily cap of 50 or 1000 depending on credits).

```jsonc
// fusion_query(question: string, force?: boolean, panel_size?: 1|2|3, preset?: "quality"|"budget")
```
