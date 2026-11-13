"""Shared offline fixtures for controllers, recipe plans and both storage backends."""
import json

import pytest

from manager.ai_manager import AIManager, StubAIProvider
from manager.controller import CaloraController
from manager.data_manager import JsonDataManager, SqliteDataManager
from manager.job_manager import PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import MealCandidate
from tests.helpers import make_profile, make_request, sample_response


@pytest.fixture
def response():
    return sample_response()


@pytest.fixture
def response_text(response):
    return json.dumps(response)


@pytest.fixture
def candidates(response):
    return [MealCandidate.from_dict(item) for item in response["recommendations"]]


@pytest.fixture
def request_both():
    return make_request(make_profile())


@pytest.fixture(params=["json", "sqlite"])
def data_manager(request, tmp_path):
    if request.param == "sqlite":
        return SqliteDataManager(str(tmp_path / "calora.db"))
    return JsonDataManager(str(tmp_path / "users.json"), str(tmp_path / "plans.json"))


@pytest.fixture
def controller_factory(tmp_path):
    pools = []

    def create(responses, data=None):
        index = len(pools)
        jobs = PlanJobManager(max_workers=2)
        pools.append(jobs)
        provider = StubAIProvider(responses)
        backend = data if data is not None else JsonDataManager(
            str(tmp_path / f"users-{index}.json"), str(tmp_path / f"plans-{index}.json"))
        controller = CaloraController(AIManager(provider, retry_delay=0), LogicManager(), backend, jobs)
        return controller, provider

    yield create
    for jobs in pools:
        jobs.shutdown()
