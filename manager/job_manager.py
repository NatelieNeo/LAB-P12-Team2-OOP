"""Background plan generation (ADVANCED FEATURE: threading).

Generating a plan waits several seconds on the AI. The web UI hands the work to
a thread pool and polls for the result; the CLI shows a spinner thread."""

from enum import Enum
import logging
import threading
from typing import Any, Optional

from manager.exceptions import CaloraError
from manager.models import MealPlan


logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class PlanJob:
    """One background task. Worker and request threads both touch it, so its
    state changes happen under a lock."""

    def __init__(self, job_id: str, owner: str, context: Any = None):
        self._job_id = job_id
        self._owner = owner
        self._context = context
        self._status = JobStatus.PENDING
        self._result: Optional[MealPlan] = None
        self._error: Optional[CaloraError] = None
        self._lock = threading.Lock()

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def owner(self) -> str:
        return self._owner

    @property
    def context(self) -> Any:
        """What the job was started for (the MealPlanRequest), kept for the result page."""
        return self._context

    @property
    def status(self) -> JobStatus:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[MealPlan]:
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[CaloraError]:
        with self._lock:
            return self._error

    def mark_running(self) -> None:
        with self._lock:
            self._status = JobStatus.RUNNING

    def complete(self, result: MealPlan) -> None:
        with self._lock:
            self._result = result
            self._status = JobStatus.DONE

    def fail(self, error: CaloraError) -> None:
        with self._lock:
            self._error = error
            self._status = JobStatus.FAILED