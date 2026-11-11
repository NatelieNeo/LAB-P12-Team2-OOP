"""ADVANCED FEATURE (threading): ask several AI models at the same time.

ParallelAIProvider is itself an AIProvider (Strategy pattern), so AIManager
uses it exactly like a single model and needs no change (OCP, LSP). It starts
one thread per model and returns the first response that passes validation.
If a model fails, times out or answers with something invalid, the others are
the safety net."""
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Sequence

from manager.ai_manager import AIProvider
from manager.exceptions import AIConnectionError, AIResponseError, AllProvidersFailedError

logger = logging.getLogger(__name__)


class ParallelAIProvider(AIProvider):
    """Races several providers; the first valid answer wins.

    Providers listed earlier take priority when two answers arrive together.
    A request that is still running when another provider wins cannot be
    cancelled; its result is simply ignored."""

    def __init__(self, providers: Sequence[AIProvider],
                validator: Optional[Callable[[str], bool]] = None,
                timeout: float = 60.0):
        if not providers:
            raise ValueError("ParallelAIProvider needs at least one provider.")
        self._providers: List[AIProvider] = list(providers)
        self._validator = validator
        self._timeout = timeout
        self._winner: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "parallel(" + ", ".join(provider.name for provider in self._providers) + ")"

    @property
    def last_winner(self) -> Optional[str]:
        """Name of the provider whose answer was used last (None before the first call)."""
        with self._lock:
            return self._winner

    def _ask(self, provider: AIProvider, prompt: str) -> str:
        """Runs in a worker thread: one provider's answer, checked for validity."""
        text = provider.generate(prompt)
        if self._validator is not None and not self._validator(text):
            raise AIResponseError(f"{provider.name} returned an invalid response.")
        return text

    def generate(self, prompt: str) -> str:
        """Return the first valid response, or raise if every provider fails.

        Raises AllProvidersFailedError (an AIConnectionError) if at least one
        provider had a connection problem or timed out, and AIResponseError if
        every provider answered but none answered validly."""
        # Not used as a context manager: leaving a `with` block would wait for the slower threads.
        executor = ThreadPoolExecutor(max_workers=len(self._providers), thread_name_prefix="ai-provider")
        futures: Dict[Future, AIProvider] = {
            executor.submit(self._ask, provider, prompt): provider for provider in self._providers}
        failures: Dict[str, Exception] = {}
        deadline = time.monotonic() + self._timeout
        pending = set(futures)
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    break
                for future in sorted(done, key=lambda f: self._providers.index(futures[f])):
                    provider = futures[future]
                    try:
                        text = future.result()
                    except Exception as error:  # noqa: BLE001 - any failure just means "try the others"
                        failures[provider.name] = error
                        logger.warning("%s failed: %s", provider.name, error)
                        continue
                    with self._lock:
                        self._winner = provider.name
                    logger.info("Using the response from %s.", provider.name)
                    return text
            for future in pending:
                provider = futures[future]
                failures[provider.name] = AIConnectionError(
                    f"{provider.name} did not answer within {self._timeout:g} seconds.")
                logger.warning("%s timed out.", provider.name)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if all(isinstance(error, AIResponseError) for error in failures.values()):
            raise AIResponseError("No AI model returned a valid response: " + self._describe(failures))
        raise AllProvidersFailedError("All AI models failed: " + self._describe(failures), failures)

    @staticmethod
    def _describe(failures: Dict[str, Exception]) -> str:
        return "; ".join(f"{name}: {error}" for name, error in failures.items())
