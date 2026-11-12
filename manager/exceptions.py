"""Exception hierarchy for Calora (see the exception matrix in the report)."""


class CaloraError(Exception):
    """Base class for every error Calora raises on purpose."""


class InvalidInputError(CaloraError):
    """User input failed validation (raised by UserInputValidator)."""


class AIConnectionError(CaloraError):
    """The AI service could not be reached or is not configured."""

    def __init__(self, message, retryable=True):
        super().__init__(message)
        self.retryable = retryable


class AIResponseError(CaloraError):
    """The AI answered, but the response was empty, malformed or invalid."""


class StorageError(CaloraError):
    """A data file or database could not be read or written."""


class NoSuitableMealsError(CaloraError):
    """No meal survived the filters, so a plan cannot be built."""


class AllProvidersFailedError(AIConnectionError):
    """Every AI provider in a parallel run failed, timed out or returned an
    invalid response. ``failures`` maps each provider's name to its error.

    It is retryable if at least one failure was retryable, so the AIManager
    only gives up immediately when no provider could ever succeed (for
    example, no API key is configured)."""

    def __init__(self, message, failures=None):
        failures = dict(failures or {})
        retryable = any(getattr(error, "retryable", True) for error in failures.values()) if failures else True
        super().__init__(message, retryable=retryable)
        self.failures = failures
