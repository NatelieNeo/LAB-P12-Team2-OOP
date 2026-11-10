"""Background plan generation (ADVANCED FEATURE: threading).

Generating a plan waits several seconds on the AI. The web UI hands the work to
a thread pool and polls for the result; the CLI shows a spinner thread."""

import logging

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
