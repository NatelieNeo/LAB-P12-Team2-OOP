"""Exercise both storage backends with real temporary files and databases."""
import json
import sqlite3
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


def test_delete_plan_removes_only_the_plan_and_reports_missing_records(data_manager):
    profile = make_profile()
    data_manager.save_profile(profile)
    plan = LogicManager().build_meal_plan(
        [MealCandidate.from_dict(meal())], make_request(profile))
    data_manager.save_plan(plan, profile.username)
    plan_id = next(iter(data_manager.plans_for_user(profile.username)))

    assert data_manager.delete_plan(plan_id)
    assert not data_manager.plans_for_user(profile.username)
    assert data_manager.get_profile(profile.username) is not None
    assert not data_manager.delete_plan(plan_id)


def test_delete_profile_removes_profile_and_its_plans(data_manager):
    profile = make_profile()
    data_manager.save_profile(profile)
    plan = LogicManager().build_meal_plan(
        [MealCandidate.from_dict(meal())], make_request(profile))
    data_manager.save_plan(plan, profile.username)

    assert data_manager.delete_profile(profile.username)
    assert data_manager.get_profile(profile.username) is None
    assert not data_manager.plans_for_user(profile.username)
    assert not data_manager.delete_profile(profile.username)


def test_sqlite_plan_delete_cascades_to_meal_details(tmp_path):
    db_path = tmp_path / "plan-cascade.db"
    manager = SqliteDataManager(str(db_path))
    profile = make_profile()
    manager.save_profile(profile)
    plan = LogicManager().build_meal_plan(
        [MealCandidate.from_dict(meal())], make_request(profile))
    manager.save_plan(plan, profile.username)
    plan_id = next(iter(manager.plans_for_user(profile.username)))

    queries = {
        "meal_plans": "SELECT COUNT(*) FROM meal_plans WHERE plan_id = ?",
        "missing_slots": "SELECT COUNT(*) FROM meal_plan_missing_slots WHERE plan_id = ?",
        "meals": "SELECT COUNT(*) FROM meals WHERE plan_id = ?",
        "ingredients": (
            "SELECT COUNT(*) FROM meal_ingredients "
            "WHERE meal_id IN (SELECT meal_id FROM meals WHERE plan_id = ?)"),
        "recipes": (
            "SELECT COUNT(*) FROM meal_recipes "
            "WHERE meal_id IN (SELECT meal_id FROM meals WHERE plan_id = ?)"),
        "recipe_steps": (
            "SELECT COUNT(*) FROM meal_recipe_steps "
            "WHERE meal_id IN (SELECT meal_id FROM meals WHERE plan_id = ?)"),
    }

    def counts():
        connection = sqlite3.connect(str(db_path))
        try:
            return {
                name: connection.execute(query, (plan_id,)).fetchone()[0]
                for name, query in queries.items()
            }
        finally:
            connection.close()

    before = counts()
    assert before["meal_plans"] == 1
    assert all(before[name] > 0 for name in (
        "meals", "ingredients", "recipes", "recipe_steps"))

    assert manager.delete_plan(plan_id)
    assert all(count == 0 for count in counts().values())
    assert manager.get_profile(profile.username) is not None


def test_sqlite_profile_delete_cascades_through_related_records(tmp_path):
    db_path = tmp_path / "cascade.db"
    manager = SqliteDataManager(str(db_path))
    profile = make_profile(dietary_restrictions=["peanut"])
    manager.save_profile(profile)
    manager.update_profile(profile.username, {"pantry_ingredients": ["rice"]})
    plan = LogicManager().build_meal_plan(
        [MealCandidate.from_dict(meal())], make_request(profile))
    manager.save_plan(plan, profile.username)

    related_tables = (
        "user_dietary_restrictions",
        "user_pantry_ingredients",
        "meal_plans",
        "meal_plan_missing_slots",
        "meals",
        "meal_ingredients",
        "meal_recipes",
        "meal_recipe_steps",
    )
    connection = sqlite3.connect(str(db_path))
    try:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in related_tables
        }
    finally:
        connection.close()
    assert all(before[table] > 0 for table in (
        "user_dietary_restrictions",
        "user_pantry_ingredients",
        "meal_plans",
        "meals",
        "meal_ingredients",
        "meal_recipes",
        "meal_recipe_steps",
    ))

    assert manager.delete_profile(profile.username)

    connection = sqlite3.connect(str(db_path))
    try:
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in related_tables
        }
    finally:
        connection.close()
    assert all(count == 0 for count in after.values())


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
