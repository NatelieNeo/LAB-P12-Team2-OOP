"""Exercise both storage backends with real temporary files and databases."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import config
from manager.controller import AppFactory
from manager.data_manager import JsonDataManager, SqliteDataManager
from manager.exceptions import InvalidInputError
from manager.logic_manager import LogicManager, RecipeBuilder
from manager.models import MealCandidate
from tests.helpers import make_profile, make_request, meal


def test_profile_update_preserves_preferences_and_reports_missing_user(data_manager):
    profile = make_profile(preferences={"meal_preference": "rice", "daily_budget": 12})
    data_manager.save_profile(profile)
    assert data_manager.get_profile("nisha").name == "Nisha"
    assert data_manager.update_profile("nisha", {"age": 30, "dietary_restrictions": ["peanut"]})
    loaded = data_manager.get_profile("nisha")
    assert loaded.age == 30 and loaded.dietary_restrictions == ["peanut"]
    assert loaded.preferences["daily_budget"] == 12
    assert not data_manager.update_profile("missing-user", {"age": 30})


def test_structured_ingredient_measurements_survive_saved_recipe(data_manager):
    profile = make_profile()
    data_manager.save_profile(profile)
    candidate = MealCandidate.from_dict(meal(ingredients=[
        {"item": "rice", "quantity": 100, "unit": "g"},
        {"item": "water", "quantity": 200, "unit": "ml"}]))
    plan = LogicManager().build_meal_plan([candidate], make_request(profile))
    data_manager.save_plan(plan, profile.username)
    loaded = next(iter(data_manager.plans_for_user(profile.username).values()))
    recipe = RecipeBuilder().build(loaded)[0]
    assert recipe.ingredients == ["100 g rice", "200 ml water"]
    assert recipe.steps == candidate.recipe_steps
    assert recipe.portion_size == "1 serving"


def test_parallel_plan_saves_keep_every_record(data_manager):
    profile = make_profile()
    data_manager.save_profile(profile)
    plan = LogicManager().build_meal_plan([MealCandidate.from_dict(meal())], make_request(profile))
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(data_manager.save_plan, plan, profile.username) for _ in range(12)]
        for future in futures:
            future.result(timeout=10)
    assert len(data_manager.plans_for_user(profile.username)) == 12


def test_history_does_not_leak_between_users(data_manager):
    first, second = make_profile(), make_profile(username="other-user")
    for profile in (first, second):
        data_manager.save_profile(profile)
    plan = LogicManager().build_meal_plan([MealCandidate.from_dict(meal())], make_request(first))
    data_manager.save_plan(plan, first.username)
    assert len(data_manager.plans_for_user(first.username)) == 1
    assert not data_manager.plans_for_user(second.username)


@pytest.mark.parametrize("backend,manager_type", [("json", JsonDataManager), ("sqlite", SqliteDataManager)])
def test_factory_honours_configured_backend_without_touching_real_data(monkeypatch, tmp_path, backend, manager_type):
    monkeypatch.setenv("CALORA_STORAGE", backend)
    monkeypatch.setattr(config, "USER_FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr(config, "MEAL_PLAN_FILE", str(tmp_path / "plans.json"))
    monkeypatch.setattr(config, "SQLITE_DB_FILE", str(tmp_path / "calora.db"))
    assert isinstance(AppFactory.build_data_manager(), manager_type)


def test_factory_rejects_an_unknown_backend():
    with pytest.raises(InvalidInputError, match="CALORA_STORAGE"):
        AppFactory.build_data_manager("typo")


def test_json_corruption_is_preserved_before_new_records_are_written(tmp_path):
    path = tmp_path / "users.json"
    path.write_text("broken JSON", encoding="utf-8")
    manager = JsonDataManager(str(path), str(tmp_path / "plans.json"))
    manager.save_profile(make_profile())
    assert path.with_suffix(".json.corrupt").read_text(encoding="utf-8") == "broken JSON"
    assert next(iter(json.loads(path.read_text(encoding="utf-8")).values()))["username"] == "nisha"
