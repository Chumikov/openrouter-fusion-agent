"""Multi-model deliberation agent for OpenRouter Fusion on free models.

Provides a CLI and an MCP server that expose OpenRouter's ``openrouter:fusion``
server tool backed entirely by free (``:free``) models, with budget-aware
safeguards against rate limits and negative balances.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
