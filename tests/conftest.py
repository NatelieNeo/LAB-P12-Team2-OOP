"""Shared sample data. Nothing here needs the network, an API key or real data files."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manager.ai_manager import AIManager, StubAIProvider  # noqa: E402
from manager.controller import CaloraController  # noqa: E402
from manager.data_manager import JsonDataManager, SqliteDataManager  # noqa: E402
from manager.job_manager import PlanJobManager  # noqa: E402
from manager.logic_manager import LogicManager  # noqa: E402
from manager.models import MealCandidate, MealPlanRequest, UserProfile  # noqa: E402


def meal(slot="Breakfast", name="Eggs on Toast", calories=400, cost=2.5, **changes):
    """A Gemini-style recommendation with exactly the expected fields."""
    result = {
        "meal_type": slot, "meal_name": name, "meal_source_type": "home_cooked",
        "calories": calories, "protein_g": 25, "carbs_g": 40, "fat_g": 15,
        "estimated_cost": cost, "portion_size": "1 serving",
        "pantry_ingredients_used": ["eggs"],
        "ingredients": ["2 eggs", "2 slices wholemeal bread"],
        "recipe_steps": ["Boil eggs for 10 minutes.", "Toast bread and serve."],
    }
    result.update(changes)
    return result


def eat_out(slot, name, calories, cost, **changes):
    return meal(slot, name, calories, cost, meal_source_type="eat_out",
                pantry_ingredients_used=[], ingredients=[], recipe_steps=[], **changes)


def sample_response():
    """Ten realistic options, exactly as the AI is asked to return them."""
    return {
        "estimated_nutrition_targets": {
            "calorie_target": 1650, "protein_target_g": 90, "carb_target_g": 185, "fat_target_g": 55},
        "recommendations": [
            meal(),
            eat_out("Lunch", "Chicken Rice", 500, 4.5),
            meal("Dinner", "Chicken and Rice Bowl", 520, 3.5,
                 pantry_ingredients_used=["chicken", "rice"],
                 ingredients=["150g chicken breast", "1 cup cooked rice"],
                 recipe_steps=["Cook chicken thoroughly.", "Serve with rice."]),
            meal("Breakfast", "Boiled Eggs and Toast", 250, 1.2),
            meal("Lunch", "Garlic Chicken Pasta", 480, 3),
            eat_out("Dinner", "Yong Tau Foo", 420, 6),
            eat_out("Lunch", "Sliced Fish Soup", 380, 5.5),
            meal("Dinner", "Salmon Rice Bowl", 600, 7),
            eat_out("Breakfast", "Kaya Toast Set", 350, 3.5),
            meal("Lunch", "Vegetable Rice Bowl", 650, 4),
        ],
    }


@pytest.fixture
def response():
    return sample_response()


@pytest.fixture
def response_text(response):
    return json.dumps(response)


@pytest.fixture
def profile():
    return UserProfile("nisha", "Nisha", 21, "female", 52, 165, "sedentary", "maintain", [])


@pytest.fixture
def request_both(profile):
    return MealPlanRequest(profile, "rice dishes", "both", ["eggs", "chicken", "rice"], 10.5)


@pytest.fixture
def candidates(response):
    return [MealCandidate.from_dict(item) for item in response["recommendations"]]


def make_request(profile, **overrides):
    values = dict(meal_preference="rice dishes", meal_source="both",
                  pantry_ingredients=["eggs", "chicken", "rice"], daily_budget=10.5)
    values.update(overrides)
    return MealPlanRequest(profile, **values)


def make_profile(**overrides):
    values = dict(username="nisha", name="Nisha", age=21, gender="female", weight_kg=52,
                  height_cm=165, activity_level="sedentary", goal="maintain", dietary_restrictions=[])
    values.update(overrides)
    return UserProfile(**values)


@pytest.fixture(params=["json", "sqlite"])
def data_manager(request, tmp_path):
    """Every storage test runs against both backends (they must be interchangeable)."""
    if request.param == "json":
        return JsonDataManager(str(tmp_path / "users.json"), str(tmp_path / "plans.json"))
    return SqliteDataManager(str(tmp_path / "calora.db"))


@pytest.fixture
def controller_factory(tmp_path):
    """Builds a controller wired to the offline stub provider and temp storage."""
    created = []

    def build(responses, data=None, max_retries=3):
        jobs = PlanJobManager(max_workers=2)
        created.append(jobs)
        provider = StubAIProvider(responses)
        ai = AIManager(provider, max_retries=max_retries, retry_delay=0)
        data = data or JsonDataManager(str(tmp_path / "u.json"), str(tmp_path / "p.json"))
        return CaloraController(ai, LogicManager(), data, jobs), provider

    yield build
    for jobs in created:
        jobs.shutdown()
