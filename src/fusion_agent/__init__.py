"""Multi-model deliberation agent for OpenRouter Fusion on free models.

Provides a CLI and an MCP server that expose OpenRouter's ``openrouter:fusion``
server tool backed entirely by free (``:free``) models, with budget-aware
safeguards against rate limits and negative balances.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("openrouter-fusion-agent")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0+unknown"
