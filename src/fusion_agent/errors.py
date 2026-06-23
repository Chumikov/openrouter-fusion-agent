"""Exception hierarchy for the fusion agent."""

from __future__ import annotations


class FusionError(RuntimeError):
    """Base class for all fusion-agent errors."""


class FusionConfigError(FusionError):
    """Raised on misconfiguration (e.g. a missing API key)."""


class FusionBudgetError(FusionError):
    """Raised when the free-model budget (RPD/RPM) or balance would be exceeded."""


class FusionAPIError(FusionError):
    """Raised on an unrecoverable upstream OpenRouter error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        body: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.error_type = error_type
