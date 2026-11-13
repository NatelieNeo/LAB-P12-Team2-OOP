"""CaloraController end to end with the offline stub provider."""
import json
import time

import pytest

from manager.data_manager import SqliteDataManager
from manager.exceptions import AIConnectionError, AIResponseError, NoSuitableMealsError
from manager.job_manager import JobStatus
from manager.models import MealPlan
from tests.conftest import make_profile, make_request, meal


def test_create_meal_plan_saves_plan_and_preferences(controller_factory, response_text):
    controller, provider = controller_factory(response_text)
    profile = make_profile()
    controller.register_profile(profile)
    plan = controller.create_meal_plan(make_request(profile))
    assert isinstance(plan, MealPlan) and plan.is_complete
    history = controller.get_history("nisha")
    assert len(history) == 1 and next(iter(history.values())).to_dict() == plan.to_dict()
    assert controller.get_profile("nisha").preferences["meal_preference"] == "rice dishes"
    assert provider.call_count == 1


def test_shopping_list_comes_from_the_requests_pantry(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    profile = make_profile()
    request = make_request(profile)
    plan = controller.create_meal_plan(request)
    assert controller.get_shopping_list(plan, request).items == ["2 slices wholemeal bread"]


def test_ai_failure_raises_and_saves_nothing(controller_factory):
    controller, _ = controller_factory("not json")
    profile = make_profile()
    controller.register_profile(profile)
    with pytest.raises(AIResponseError):
        controller.create_meal_plan(make_request(profile))
    assert controller.get_history("nisha") == {}


def test_connection_failure_is_reported(controller_factory):
    controller, _ = controller_factory(AIConnectionError("down"))
    with pytest.raises(AIConnectionError):
        controller.create_meal_plan(make_request(make_profile()))


def test_everything_filtered_out_raises_no_suitable_meals(controller_factory, response):
    response["recommendations"] = [meal(name="Peanut Toast", ingredients=["peanut"]) for _ in range(10)]
    controller, _ = controller_factory(json.dumps(response))
    request = make_request(make_profile(dietary_restrictions=["peanut"]))
    with pytest.raises(NoSuitableMealsError):
        controller.create_meal_plan(request)


def test_profile_register_find_and_update(controller_factory):
    controller, _ = controller_factory("")
    assert controller.get_profile("nisha") is None
    controller.register_profile(make_profile())
    assert controller.get_profile("  nisha ").name == "Nisha"
    assert controller.update_profile("nisha", make_profile(age=40)) is True
    assert controller.get_profile("nisha").age == 40


def test_background_job_produces_the_same_plan(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    profile = make_profile()
    controller.register_profile(profile)
    request = make_request(profile)
    job = controller.get_plan_job(controller.submit_plan_job(request))
    deadline = time.monotonic() + 5
    while job.status not in (JobStatus.DONE, JobStatus.FAILED) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status is JobStatus.DONE
    assert job.context is request and job.owner == "nisha"
    assert len(controller.get_history("nisha")) == 1


def test_failed_background_job_carries_the_error(controller_factory):
    controller, _ = controller_factory("not json")
    job = controller.get_plan_job(controller.submit_plan_job(make_request(make_profile())))
    deadline = time.monotonic() + 5
    while job.status in (JobStatus.PENDING, JobStatus.RUNNING) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status is JobStatus.FAILED and isinstance(job.error, AIResponseError)


def test_works_the_same_on_sqlite(controller_factory, response_text, tmp_path):
    controller, _ = controller_factory(response_text, data=SqliteDataManager(str(tmp_path / "c.db")))
    profile = make_profile()
    controller.register_profile(profile)
    plan = controller.create_meal_plan(make_request(profile))
    assert next(iter(controller.get_history("nisha").values())).to_dict() == plan.to_dict()


def test_ai_is_asked_once_per_plan_even_with_a_retry(controller_factory, response_text):
    controller, provider = controller_factory(["bad", response_text])
    controller.create_meal_plan(make_request(make_profile()))
    assert provider.call_count == 2
