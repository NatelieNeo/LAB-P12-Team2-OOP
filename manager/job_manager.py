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


class PlanJobManager:
    """Owns the thread pool and the registry of jobs (composition)."""

    def __init__(self, max_workers: int = PLAN_WORKER_THREADS, max_jobs: int = MAX_TRACKED_JOBS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="calora-plan")
        self._jobs: "OrderedDict[str, PlanJob]" = OrderedDict()
        self._max_jobs = max_jobs
        self._lock = threading.Lock()

    def submit(self, task: Callable[[], MealPlan], owner: str = "", context: Any = None) -> str:
        """Start `task` in a worker thread and return a job id straight away."""
        job = PlanJob(uuid.uuid4().hex, owner, context)
        with self._lock:
            self._jobs[job.job_id] = job
            self._prune()
        self._executor.submit(self._run, job, task)
        return job.job_id

    def get(self, job_id: str) -> Optional[PlanJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job: PlanJob, task: Callable[[], MealPlan]) -> None:
        job.mark_running()
        try:
            job.complete(task())
        except CaloraError as error:
            logger.warning("Plan job %s failed: %s", job.job_id, error)
            job.fail(error)
        except Exception as error:  # a worker must never die silently
            logger.exception("Unexpected error in plan job %s", job.job_id)
            job.fail(CaloraError(f"Unexpected error: {error}"))

    def _prune(self) -> None:
        """Forget the oldest finished jobs once too many are tracked."""
        while len(self._jobs) > self._max_jobs:
            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.DONE, JobStatus.FAILED):
                    del self._jobs[job_id]
                    break
            else:
                return

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


class ProgressSpinner(threading.Thread):
    """CLI spinner shown while a plan is being generated (inherits Thread)."""

    FRAMES = "|/-\\"

    def __init__(self, message: str = "Generating your meal plan", stream=None, interval: float = 0.15):
        super().__init__(daemon=True, name="calora-spinner")
        self._message = message
        self._stream = stream if stream is not None else sys.stdout
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            self._stream.write(f"\r{self._message}... {frame}")
            self._stream.flush()
            time.sleep(self._interval)
        self._stream.write("\r" + " " * (len(self._message) + 8) + "\r")
        self._stream.flush()

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join()
