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
from tests.helpers import FakeProvider, response_data, start_plan, wait_for_plan


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
    assert b"AI estimates" in response.data
    assert b"{'item':" not in response.data
    assert response.data.index(b'class="plan-card plan-overview"') < response.data.index(b'class="meals"') < response.data.index(b'class="plan-card shopping-card"')
    assert b"Nothing to buy" in response.data
    shopping_card = response.data.split(b'class="plan-card shopping-card"', 1)[1].split(b'</section>', 1)[0]
    assert b"Already in your pantry" not in shopping_card and b"100 g rice" not in shopping_card
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


def test_web_shopping_card_contains_only_missing_ingredients(browser):
    client, _, providers = browser
    register(client)
    for provider in providers:
        data = json.loads(provider.outcomes[0])
        for candidate in data["recommendations"]:
            candidate["ingredients"] = [
                "rice", "egg", "chicken breast", "2 green onions", "1 tbsp soy sauce",
                {"item": "soy sauce", "quantity": 2, "unit": "tbsp"}, "sesame oil (5 ml)",
                "(15 g) gochujang", "(5 ml) sesame oil", "(5 ml) cooking oil", "[5 g] garlic",
                "ginger - 10g"]
        provider.outcomes = (json.dumps(data),)
    response = client.post("/plan", data={"meal_source": "in", "ingredients": "rice, eggs, chicken"})
    assert response.status_code == 200
    shopping_card = response.data.split(b'class="plan-card shopping-card"', 1)[1].split(b'</section>', 1)[0]
    assert b"<li>green onions</li>" in shopping_card and b"<li>soy sauce</li>" in shopping_card
    assert b"<li>sesame oil</li>" in shopping_card
    for name in (b"gochujang", b"cooking oil", b"garlic", b"ginger"):
        assert b"<li>" + name + b"</li>" in shopping_card
    assert b"15 g" not in shopping_card and b"5 g" not in shopping_card and b"10g" not in shopping_card
    assert shopping_card.count(b"<li>sesame oil</li>") == 1
    assert shopping_card.count(b"<li>soy sauce</li>") == 1
    assert b"2 green onions" not in shopping_card and b"tbsp" not in shopping_card and b"5 ml" not in shopping_card
    assert b"<li>2 green onions</li>" in response.data and b"<li>1 tbsp soy sauce</li>" in response.data
    assert b"<li>(15 g) gochujang</li>" in response.data  # Recipe measurements remain available.
    assert b"<li>rice</li>" not in shopping_card
    assert b"<li>egg</li>" not in shopping_card
    assert b"<li>chicken breast</li>" not in shopping_card


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


def test_background_plan_has_recipes_and_cannot_be_read_by_another_session(browser):
    _, controller, _ = browser
    isolated_app = web.WebApp(controller, background_plans=True).flask_app
    client = isolated_app.test_client()
    register(client)
    job_id = start_plan(client, {"meal_source": "in", "ingredients": "rice", "budget": "0"})
    assert wait_for_plan(client, job_id) == "DONE"
    result = client.get(f"/plan/result/{job_id}")
    assert b"View recipe" in result.data and b"100 g rice" in result.data
    assert len(controller.get_history("web-user")) == 1
    stranger = isolated_app.test_client()
    assert stranger.get(f"/plan/status/{job_id}").status_code == 404
    assert stranger.get(f"/plan/result/{job_id}").status_code == 302


@pytest.mark.parametrize("endpoint,method", [("/profile", "get"), ("/profile/edit", "get"),
                                           ("/profile/update", "post"), ("/plan", "post")])
def test_stale_profile_session_redirects_instead_of_crashing(browser, endpoint, method):
    client, _, _ = browser
    with client.session_transaction() as session:
        session["username"] = "missing-profile"
    assert getattr(client, method)(endpoint).status_code == 302


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


def test_cli_profile_updates_and_saved_recipe_menu_are_available(browser):
    _, controller, providers = browser
    controller.register_profile(UserProfile.from_dict({**form_profile(), "gender": "female"}))
    answers = iter(["web-user", "4", "Updated CLI Name"] + [""] * 7 + ["", "3", "", "r", "", "5"])
    shown = []
    io = IOManager(input_fn=lambda prompt: next(answers), output_fn=lambda *parts: shown.append(" ".join(map(str, parts))))
    CliApp(controller, io).run()
    assert controller.get_profile("web-user").name == "Updated CLI Name"
    assert any("No recipes found" in line for line in shown)
    assert all(p.calls == 0 for p in providers)


def test_cli_stops_spinner_and_returns_to_menu_on_ai_failure(browser):
    from unittest.mock import Mock
    _, controller, providers = browser
    controller.register_profile(UserProfile.from_dict({**form_profile(), "gender": "female"}))
    for provider in providers:
        provider.outcomes = ("invalid JSON",)
    answers = iter(["web-user", "1", "rice", "in", "rice", "0", "", "5"])
    shown, spinner = [], Mock()
    io = IOManager(input_fn=lambda prompt: next(answers), output_fn=lambda *parts: shown.append(" ".join(map(str, parts))))
    CliApp(controller, io, spinner_factory=lambda: spinner).run()
    spinner.start.assert_called_once()
    spinner.stop.assert_called_once()
    assert any("Unable to complete request" in line for line in shown)
    assert not controller.get_history("web-user")


def test_importing_entry_points_does_not_create_database_or_call_ai():
    # The browser fixture is deliberately absent; entry points can be imported
    # for Gunicorn/testing before any credentials or storage are provisioned.
    result = subprocess.run([sys.executable, "-B", "-c",
                             "import app, main; assert 'calora_controller' not in app.app.extensions"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
