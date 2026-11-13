"""Shared deterministic providers, sample data and UI drivers; no live API calls."""
import re
import threading
import time

from manager.ai_manager import AIProvider
from manager.logic_manager import NutritionCalculator
from manager.models import MealPlanRequest, UserProfile


class FakeProvider(AIProvider):
    """Record calls and replay outcomes, with optional overlap barriers/gates."""

    def __init__(self, name, *outcomes, barrier=None, gate=None):
        self._name, self.outcomes = name, outcomes
        self.barrier, self.gate = barrier, gate
        self.calls, self.prompts, self.threads = 0, [], []
        self.started = threading.Event()
        self._lock = threading.Lock()

    @property
    def name(self):
        return self._name

    def generate(self, prompt):
        with self._lock:
            outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
            self.calls += 1
            self.prompts.append(prompt)
            self.threads.append(threading.current_thread().name)
        self.started.set()
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if self.gate is not None and not self.gate.wait(timeout=5):
            raise TimeoutError("Test gate was not released")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_profile(**changes):
    data = dict(username="nisha", name="Nisha", age=21, gender="female", weight_kg=52,
                height_cm=165, activity_level="sedentary", goal="maintain", dietary_restrictions=[])
    data.update(changes)
    return UserProfile(**data)


def make_request(profile, **changes):
    data = dict(meal_preference="rice dishes", meal_source="both",
                pantry_ingredients=["eggs", "chicken", "rice"], daily_budget=10.5)
    data.update(changes)
    return MealPlanRequest(profile, **data)


def meal(slot="breakfast", name="Eggs on Toast", calories=400, cost=2.5,
         source="home_cooked", **changes):
    home = source == "home_cooked"
    data = dict(meal_type=slot.lower(), meal_name=name, meal_source_type=source,
                calories=calories, protein_g=25, carbs_g=40, fat_g=15, estimated_cost=cost,
                portion_size="1 serving", pantry_ingredients_used=["eggs"] if home else [],
                ingredients=["2 eggs", "2 slices wholemeal bread"] if home else ["rice", "chicken"],
                recipe_steps=["Boil eggs for 10 minutes.", "Toast bread and serve."] if home else [])
    data.update(changes)
    return data


def eat_out(slot, name, calories, cost):
    return meal(slot, name, calories, cost, source="eat_out")


def sample_response():
    """A mixed-source set whose best day's only missing pantry item is bread."""
    return {
        "estimated_nutrition_targets": {
            "calorie_target": 1650, "protein_target_g": 90, "carb_target_g": 185, "fat_target_g": 55},
        "recommendations": [
            meal(), meal(name="Boiled Eggs and Toast", calories=250, cost=1.2),
            eat_out("breakfast", "Kaya Toast Set", 350, 3.5),
            eat_out("lunch", "Chicken Rice", 500, 4.5),
            meal("lunch", "Garlic Chicken Pasta", 480, 3),
            eat_out("lunch", "Sliced Fish Soup", 380, 5.5),
            meal("lunch", "Vegetable Rice Bowl", 650, 4),
            meal("dinner", "Chicken and Rice Bowl", 520, 3.5, ingredients=["chicken", "rice"]),
            eat_out("dinner", "Yong Tau Foo", 420, 6),
            meal("dinner", "Salmon Rice Bowl", 600, 7),
        ]}


def response_data(request, calorie_factor=1.0, cost=3.0, prefix="Rice"):
    """Generate ten options at calculated targets for controlled scoring tests."""
    targets = NutritionCalculator().targets(request.profile)
    recommendations = []
    for i, slot in enumerate(("breakfast",) * 3 + ("lunch",) * 4 + ("dinner",) * 3):
        recommendations.append({
            "meal_type": slot, "meal_name": f"{prefix} {chr(65 + i)}", "meal_source_type": "home_cooked",
            "calories": targets.per_meal_calories * calorie_factor,
            **targets.per_meal_macros, "estimated_cost": cost, "portion_size": "1 serving",
            "pantry_ingredients_used": ["rice"],
            "ingredients": [{"item": "rice", "quantity": 100, "unit": "g"}],
            "recipe_steps": ["Cook rice for 15 minutes."]})
    return {"estimated_nutrition_targets": {
        "calorie_target": targets.daily_calories, "protein_target_g": targets.daily_macros["protein_g"],
        "carb_target_g": targets.daily_macros["carbs_g"], "fat_target_g": targets.daily_macros["fat_g"]},
        "recommendations": recommendations}


NEW_USER = ["nisha", "Nisha", "21", "F", "52", "165", "sedentary", "maintain", "nil"]
PROFILE_FORM = dict(username="nisha", name="Nisha", age="21", gender="F", weight_kg="52",
                    height_cm="165", activity_level="sedentary", goal="maintain", dietary_restrictions="nil")
PLAN_FORM = dict(meal_preference="rice dishes", meal_source="both", ingredients="eggs, chicken, rice", budget="0")


class QuietSpinner:
    """CLI tests use no terminal thread or animation."""

    def start(self):
        pass

    def stop(self):
        pass


def run_cli(controller, answers):
    from main import CliApp
    from manager.io_manager import IOManager
    remaining, shown = iter(answers), []
    io = IOManager(input_fn=lambda prompt: next(remaining),
                   output_fn=lambda *parts: shown.append(" ".join(map(str, parts))))
    CliApp(controller, io, spinner_factory=QuietSpinner).run()
    return "\n".join(shown)


def sign_up(client, form=None):
    response = client.post("/profile/create", data=form or PROFILE_FORM)
    assert response.status_code == 302


def start_plan(client, form=None):
    response = client.post("/plan", data=form or PLAN_FORM)
    assert response.status_code == 200
    match = re.search(r'/plan/status/([a-f0-9]+)', response.get_data(as_text=True))
    assert match, "Expected a background-job waiting page"
    return match.group(1)


def wait_for_plan(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/plan/status/{job_id}")
        assert response.status_code == 200
        status = response.get_json()["status"]
        if status in ("DONE", "FAILED"):
            return status
        time.sleep(0.01)
    raise AssertionError("Plan job did not finish")
