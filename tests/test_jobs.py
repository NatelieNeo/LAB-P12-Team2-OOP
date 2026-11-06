"""ADVANCED FEATURE: background plan jobs and the CLI spinner (threading)."""
import io
import threading
import time

import pytest

from manager.exceptions import AIConnectionError, CaloraError
from manager.job_manager import JobStatus, PlanJob, PlanJobManager, ProgressSpinner
from manager.models import MealPlan


def wait_for(job, statuses=(JobStatus.DONE, JobStatus.FAILED), timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job.status in statuses:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job stuck in {job.status}")


@pytest.fixture
def jobs():
    manager = PlanJobManager(max_workers=3)
    yield manager
    manager.shutdown()


def test_job_lifecycle():
    job = PlanJob("1", "tim")
    assert job.status is JobStatus.PENDING and job.result is None
    job.mark_running()
    assert job.status is JobStatus.RUNNING
    plan = MealPlan(1500)
    job.complete(plan)
    assert job.status is JobStatus.DONE and job.result is plan
    failing = PlanJob("2", "tim")
    failing.fail(CaloraError("boom"))
    assert failing.status is JobStatus.FAILED and str(failing.error) == "boom"


def test_submit_returns_immediately_while_work_continues(jobs):
    release = threading.Event()
    job_id = jobs.submit(lambda: (release.wait(5), MealPlan(1))[1], owner="tim")
    job = jobs.get(job_id)
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)
    release.set()
    assert wait_for(job) is JobStatus.DONE
    assert isinstance(job.result, MealPlan) and job.owner == "tim"


def test_calora_errors_become_failed_jobs(jobs):
    def task():
        raise AIConnectionError("down")
    job = jobs.get(jobs.submit(task))
    assert wait_for(job) is JobStatus.FAILED
    assert isinstance(job.error, AIConnectionError)


def test_unexpected_errors_are_contained(jobs):
    def task():
        raise ZeroDivisionError("oops")
    job = jobs.get(jobs.submit(task))
    assert wait_for(job) is JobStatus.FAILED
    assert "oops" in str(job.error)


def test_context_is_kept_for_the_result_page(jobs):
    job = jobs.get(jobs.submit(lambda: MealPlan(1), owner="tim", context={"pantry": ["rice"]}))
    assert job.context == {"pantry": ["rice"]}


def test_unknown_job_id_returns_none(jobs):
    assert jobs.get("missing") is None


def test_jobs_really_run_in_parallel(jobs):
    """Three tasks that each wait for the other two can only finish if they overlap."""
    barrier = threading.Barrier(3, timeout=5)

    def task():
        barrier.wait()
        return MealPlan(1)

    submitted = [jobs.get(jobs.submit(task)) for _ in range(3)]
    assert all(wait_for(job) is JobStatus.DONE for job in submitted)


def test_runs_on_a_worker_thread_not_the_caller(jobs):
    seen = []
    job = jobs.get(jobs.submit(lambda: (seen.append(threading.current_thread().name), MealPlan(1))[1]))
    wait_for(job)
    assert seen[0].startswith("calora-plan") and seen[0] != threading.current_thread().name


def test_finished_jobs_are_pruned():
    manager = PlanJobManager(max_workers=2, max_jobs=3)
    try:
        ids = []
        for _ in range(8):
            job_id = manager.submit(lambda: MealPlan(1))
            wait_for(manager.get(job_id))
            ids.append(job_id)
        assert manager.get(ids[0]) is None and manager.get(ids[-1]) is not None
    finally:
        manager.shutdown()


def test_spinner_is_a_thread_that_starts_and_stops():
    stream = io.StringIO()
    spinner = ProgressSpinner("Working", stream=stream, interval=0.01)
    assert isinstance(spinner, threading.Thread)
    spinner.start()
    time.sleep(0.05)
    spinner.stop()
    assert not spinner.is_alive()
    assert "Working" in stream.getvalue()


def test_spinner_stop_is_safe_when_never_started():
    ProgressSpinner(stream=io.StringIO()).stop()
