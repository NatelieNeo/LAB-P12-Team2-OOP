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
