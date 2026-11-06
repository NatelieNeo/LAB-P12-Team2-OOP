"""Both storage backends must behave identically (Liskov substitution)."""
import json
import threading

import pytest

from manager.data_manager import DataManager, JsonDataManager, SqliteDataManager
from manager.exceptions import StorageError
from manager.logic_manager import LogicManager
from manager.models import MealPlan
from tests.conftest import make_profile


def a_plan(candidates, request_both):
    return LogicManager().build_meal_plan(candidates, request_both)


def test_profile_save_find_and_exists(data_manager):
    assert not data_manager.profile_exists("nisha")
    data_manager.save_profile(make_profile())
    assert data_manager.profile_exists("nisha")
    found = data_manager.find_profiles("username", "nisha")
    assert len(found) == 1
    assert next(iter(found.values())).name == "Nisha"
    assert data_manager.find_profiles("username", "nobody") == {}
    assert data_manager.get_profile("nisha").username == "nisha"
    assert data_manager.get_profile("nobody") is None


def test_update_profile_merges_changes(data_manager):
    data_manager.save_profile(make_profile())
    assert data_manager.update_profile("nisha", {"age": 30, "goal": "gain"}) is True
    profile = data_manager.get_profile("nisha")
    assert (profile.age, profile.goal, profile.name) == (30, "gain", "Nisha")
    assert data_manager.update_profile("nobody", {"age": 1}) is False


def test_preferences_are_stored_with_the_profile(data_manager, request_both):
    data_manager.save_profile(make_profile())
    data_manager.update_profile("nisha", request_both.to_preferences())
    assert data_manager.get_profile("nisha").preferences["pantry_ingredients"] == ["eggs", "chicken", "rice"]


def test_plans_are_stored_per_user_in_order(data_manager, candidates, request_both):
    plan = a_plan(candidates, request_both)
    data_manager.save_plan(plan, "nisha")
    data_manager.save_plan(plan, "someone-else")
    data_manager.save_plan(plan, "nisha")
    stored = data_manager.plans_for_user("nisha")
    assert len(stored) == 2
    assert all(isinstance(p, MealPlan) for p in stored.values())
    assert next(iter(stored.values())).to_dict() == plan.to_dict()
    assert data_manager.plans_for_user("nobody") == {}


def test_concurrent_saves_lose_nothing(data_manager, candidates, request_both):
    plan = a_plan(candidates, request_both)
    errors = []

    def save():
        try:
            data_manager.save_plan(plan, "nisha")
            data_manager.save_profile(make_profile(username=threading.current_thread().name))
        except Exception as error:  # pragma: no cover - failure path
            errors.append(error)

    threads = [threading.Thread(target=save, name=f"user{i}") for i in range(25)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(data_manager.plans_for_user("nisha")) == 25
    assert all(data_manager.profile_exists(f"user{i}") for i in range(25))


def test_backends_are_data_managers(tmp_path):
    assert isinstance(JsonDataManager(str(tmp_path / "a"), str(tmp_path / "b")), DataManager)
    assert isinstance(SqliteDataManager(str(tmp_path / "c.db")), DataManager)


# ---- JSON specifics: missing, corrupt and unwritable files ---------------------
def test_missing_files_are_treated_as_empty(tmp_path):
    manager = JsonDataManager(str(tmp_path / "none.json"), str(tmp_path / "nope.json"))
    assert manager.find_profiles("username", "x") == {}
    assert manager.plans_for_user("x") == {}


def test_corrupt_file_is_set_aside_not_destroyed(tmp_path):
    users = tmp_path / "users.json"
    users.write_text("{this is not json", encoding="utf-8")
    manager = JsonDataManager(str(users), str(tmp_path / "plans.json"))
    assert manager.find_profiles("username", "x") == {}
    assert (tmp_path / "users.json.corrupt").read_text(encoding="utf-8") == "{this is not json"
    manager.save_profile(make_profile())
    assert manager.profile_exists("nisha")


def test_unexpected_json_shape_is_treated_as_empty(tmp_path):
    users = tmp_path / "users.json"
    users.write_text("[1, 2, 3]", encoding="utf-8")
    assert JsonDataManager(str(users), str(tmp_path / "p.json")).find_profiles("username", "x") == {}


def test_unwritable_location_raises_storage_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    manager = JsonDataManager(str(blocker / "users.json"), str(blocker / "plans.json"))
    with pytest.raises(StorageError):
        manager.save_profile(make_profile())


def test_files_are_valid_json_with_part1_shape(tmp_path, candidates, request_both):
    manager = JsonDataManager(str(tmp_path / "u.json"), str(tmp_path / "p.json"))
    manager.save_profile(make_profile())
    manager.save_plan(a_plan(candidates, request_both), "nisha")
    users = json.loads((tmp_path / "u.json").read_text(encoding="utf-8"))
    plans = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert next(iter(users.values()))["username"] == "nisha"
    assert next(iter(plans.values()))["user_id"] == "nisha"


def test_part1_data_files_are_readable(tmp_path):
    (tmp_path / "u.json").write_text(json.dumps({"abc": {
        "username": "tim", "name": "timothy", "age": 23, "gender": "male", "weight_kg": 75.0, "height_cm": 180.0,
        "activity_level": "moderate", "goal": "gain", "dietary_restrictions": ["fish"],
        "meal_preference": "savoury", "meal_source": "both", "pantry_ingredients": [], "daily_budget": 40.0}}))
    assert JsonDataManager(str(tmp_path / "u.json"), str(tmp_path / "p.json")).get_profile("tim").goal == "gain"


# ---- SQLite specifics ------------------------------------------------------------
def test_sqlite_persists_between_instances(tmp_path):
    path = str(tmp_path / "calora.db")
    SqliteDataManager(path).save_profile(make_profile())
    assert SqliteDataManager(path).profile_exists("nisha")


def test_sqlite_unusable_path_raises_storage_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(StorageError):
        SqliteDataManager(str(blocker / "calora.db"))
