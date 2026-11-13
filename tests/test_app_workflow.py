"""Exercise the existing web/CLI flows with SQLite and injected offline providers."""
import json
import subprocess
import sys

import pytest

import app as web
from main import CliApp
from manager.ai_manager import MultiAIRecommendationManager
from manager.controller import CaloraController
from manager.data_manager import SqliteDataManager
from manager.io_manager import IOManager
from manager.job_manager import PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import UserProfile
from test_multi_ai import FakeProvider, response_data


@pytest.fixture
def browser(tmp_path, monkeypatch):
    profile = UserProfile("web-user", "Web User", 24, "female", 60, 165, "sedentary", "maintain")
    from manager.models import MealPlanRequest
    text = json.dumps(response_data(MealPlanRequest(profile)))
    providers = [FakeProvider("Gemini", text), FakeProvider("Claude", text)]
    jobs = PlanJobManager()
    controller = CaloraController(MultiAIRecommendationManager(providers, max_retries=1), LogicManager(),
                                  SqliteDataManager(str(tmp_path / "web.db")), jobs)
    monkeypatch.setitem(web.app.extensions, "calora_controller", controller)
    monkeypatch.setitem(web.app.config, "TESTING", True)
    yield web.app.test_client(), controller, providers
    jobs.shutdown()


def form_profile():
    return {"username": "web-user", "name": "Web User", "age": "24", "gender": "F",
            "weight_kg": "60", "height_cm": "165", "activity_level": "sedentary",
            "goal": "maintain", "dietary_restrictions": "nil"}


def register(client):
    assert client.post("/login", data={"username": "web-user"}).status_code == 200
    assert client.post("/profile/create", data=form_profile()).status_code == 302


def test_web_profile_plan_history_update_logout(browser):
    client, controller, providers = browser
    assert client.get("/").status_code == 200
    register(client)
    assert b"Web User" in client.get("/profile").data
    response = client.post("/plan", data={"meal_preference": "rice", "meal_source": "in",
                                         "ingredients": "rice", "budget": "9", "strict_budget": "yes"})
    assert response.status_code == 200
    assert b"Selected AI:" in response.data and b"AI estimates" in response.data
    assert b"{'item':" not in response.data
    assert len(controller.get_history("web-user")) == 1  # no duplicate save
    assert all(p.calls == 1 for p in providers)
    assert providers[0].prompts == providers[1].prompts
    assert b"Rice" in client.get("/plans").data
    updated = form_profile()
    updated["name"] = "Updated Name"
    assert client.post("/profile/update", data=updated).status_code == 200
    assert controller.get_profile("web-user").name == "Updated Name"
    assert controller.get_profile("web-user").preferences["meal_preference"] == "rice"
    assert client.post("/logout").status_code == 302
    assert client.get("/plans").status_code == 302


def test_web_validation_returns_controlled_error(browser):
    client, _, _ = browser
    data = form_profile()
    data["age"] = "not-a-number"
    assert client.post("/profile/create", data=data).status_code == 400


def test_web_all_providers_failing_returns_503_and_does_not_save(browser):
    client, controller, providers = browser
    register(client)
    for provider in providers:
        provider.outcomes = ("invalid JSON",)
    assert client.post("/plan", data={"meal_source": "both"}).status_code == 503
    assert not controller.get_history("web-user")


def test_web_infeasible_strict_budget_returns_controlled_error(browser):
    client, controller, _ = browser
    register(client)
    response = client.post("/plan", data={"meal_source": "both", "budget": "6", "strict_budget": "yes"})
    assert response.status_code == 503
    assert not controller.get_history("web-user")


def test_cli_existing_user_can_generate_view_history_and_exit(browser):
    _, controller, providers = browser
    controller.register_profile(UserProfile.from_dict({**form_profile(), "gender": "female"}))
    answers = iter(["web-user", "1", "rice", "in", "rice", "9", "yes", "", "2", "", "5"])
    output = []
    io = IOManager(input_fn=lambda prompt: next(answers), output_fn=lambda *args: output.append(" ".join(map(str, args))))
    CliApp(controller, io).run()
    assert len(controller.get_history("web-user")) == 1
    assert any("Selected AI" in line for line in output)
    assert all(p.calls == 1 for p in providers)


def test_importing_entry_points_does_not_create_database_or_call_ai():
    # The browser fixture is deliberately absent; entry points can be imported
    # for Gunicorn/testing before any credentials or storage are provisioned.
    result = subprocess.run([sys.executable, "-B", "-c",
                             "import app, main; assert 'calora_controller' not in app.app.extensions"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
