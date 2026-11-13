"""Recipe feature: step-by-step instructions for home-cooked meals."""
import json

import pytest

from main import CliApp
from manager.data_manager import JsonDataManager, SqliteDataManager
from manager.io_manager import IOManager
from manager.logic_manager import LogicManager, MealTagger, NutritionCalculator, RecipeBuilder
from manager.models import MealCandidate, MealPlan, PlannedMeal, Recipe
from tests.conftest import eat_out, make_profile, make_request, meal, sample_response
from tests.test_cli import NEW_USER, QuietSpinner, run_cli
from tests.test_web import PLAN_FORM, sign_up, start_plan, wait_for_plan


def planned(slot="breakfast", source="home_cooked", steps=("Boil the eggs.", "Toast the bread."), **kw):
    values = dict(slot=slot, name="Eggs on Toast", source_type=source, calorie_range="Moderate (400-700 kcal)",
                  calorie_tag="on_target", protein_tag="High (>20-30g)", carbs_tag="High (>30-60g)",
                  fat_tag="High (>10-20g)", cost_tier="Budget (<$5)",
                  ingredients=["2 eggs", "2 slices bread"], recipe_steps=list(steps), portion_size="1 serving")
    values.update(kw)
    return PlannedMeal(**values)


def plan_with(*meals):
    plan = MealPlan(1650, 10)
    for item in meals:
        plan.add_meal(item)
    return plan


# ---- PlannedMeal keeps the steps --------------------------------------------------------
def test_planned_meal_keeps_steps_and_portion_and_round_trips():
    original = planned()
    assert original.recipe_steps == ["Boil the eggs.", "Toast the bread."]
    assert original.portion_size == "1 serving"
    copy = PlannedMeal.from_dict("breakfast", original.to_dict())
    assert copy.recipe_steps == original.recipe_steps and copy.portion_size == "1 serving"


def test_blank_steps_are_dropped_and_the_list_is_a_copy():
    item = planned(steps=["  Chop.  ", "", "   ", "Fry."])
    assert item.recipe_steps == ["Chop.", "Fry."]
    item.recipe_steps.append("tamper")
    assert item.recipe_steps == ["Chop.", "Fry."]


def test_plans_saved_before_the_recipe_feature_still_load():
    old = planned().to_dict()
    del old["recipe_steps"], old["portion_size"]
    meal_from_old_plan = PlannedMeal.from_dict("breakfast", old)
    assert meal_from_old_plan.recipe_steps == [] and meal_from_old_plan.portion_size == ""


# ---- the tagger carries steps from the AI's candidate ------------------------------------
def test_home_cooked_candidate_keeps_its_recipe(targets):
    result = MealTagger().annotate(MealCandidate.from_dict(meal()), targets)
    assert result.recipe_steps == ["Boil eggs for 10 minutes.", "Toast bread and serve."]
    assert result.portion_size == "1 serving"


def test_eat_out_candidate_has_no_recipe(targets):
    result = MealTagger().annotate(MealCandidate.from_dict(eat_out("Lunch", "Chicken Rice", 500, 4.5)), targets)
    assert result.recipe_steps == [] and result.portion_size == ""


@pytest.fixture
def targets():
    return NutritionCalculator().targets(make_profile())


# ---- Recipe ---------------------------------------------------------------------------
def test_recipe_is_read_only_and_formats_numbered_steps():
    recipe = Recipe("lunch", "Chicken Bowl", "1 bowl", ["150g chicken", "rice"], ["Cook chicken.", "Serve."])
    text = recipe.to_text()
    assert text.splitlines()[0] == "Lunch: Chicken Bowl"
    assert "Serves: 1 bowl" in text and "  - 150g chicken" in text
    assert "  1. Cook chicken." in text and "  2. Serve." in text
    recipe.steps.append("tamper")
    recipe.ingredients.append("tamper")
    assert recipe.steps == ["Cook chicken.", "Serve."] and len(recipe.ingredients) == 2
    with pytest.raises(AttributeError):
        recipe.name = "other"
    assert recipe.to_dict()["steps"] == ["Cook chicken.", "Serve."]


def test_recipe_without_steps_says_so():
    recipe = Recipe("lunch", "Old Meal")
    assert not recipe.has_steps
    assert "No cooking instructions were saved" in recipe.to_text()


# ---- RecipeBuilder ----------------------------------------------------------------------
def test_builder_makes_recipes_only_for_home_cooked_meals_in_meal_order():
    plan = plan_with(
        planned("dinner", name="Chicken Bowl"),
        planned("lunch", source="eat_out", steps=(), ingredients=[], name="Chicken Rice"),
        planned("breakfast", name="Eggs on Toast"))
    recipes = RecipeBuilder().build(plan)
    assert [(r.slot, r.name) for r in recipes] == [("breakfast", "Eggs on Toast"), ("dinner", "Chicken Bowl")]
    assert recipes[0].ingredients == ["2 eggs", "2 slices bread"]
    assert recipes[0].steps == ["Boil the eggs.", "Toast the bread."]


def test_builder_returns_nothing_for_an_all_eat_out_plan():
    plan = plan_with(planned("lunch", source="eat_out", steps=(), ingredients=[]))
    assert RecipeBuilder().build(plan) == []


def test_logic_manager_uses_the_injected_builder():
    class OneLineBuilder(RecipeBuilder):
        def build(self, plan):
            return [Recipe("lunch", "Custom", steps=["Just eat."])]

    recipes = LogicManager(recipe_builder=OneLineBuilder()).build_recipes(plan_with(planned()))
    assert [r.name for r in recipes] == ["Custom"]


def test_logic_manager_builds_recipes_from_a_real_plan(candidates, request_both):
    logic = LogicManager()
    plan = logic.build_meal_plan(candidates, request_both)
    recipes = logic.build_recipes(plan)
    assert recipes and all(r.has_steps for r in recipes)
    assert {r.name for r in recipes} == {m.name for m in plan.meals.values() if m.source_type == "home_cooked"}


# ---- controller ------------------------------------------------------------------------
def test_controller_returns_recipes_for_a_new_plan(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    profile = make_profile()
    controller.register_profile(profile)
    plan = controller.create_meal_plan(make_request(profile, meal_source="home_cooked"))
    recipes = controller.get_recipes(plan)
    assert len(recipes) == len(plan.meals) and all(r.steps for r in recipes)


# ---- storage (both backends must keep the steps) ------------------------------------------
def test_saved_plans_keep_their_recipes(data_manager):
    data_manager.save_profile(make_profile())
    data_manager.save_plan(plan_with(planned("breakfast"), planned("dinner", name="Chicken Bowl")), "nisha")
    loaded = next(iter(data_manager.plans_for_user("nisha").values()))
    recipes = RecipeBuilder().build(loaded)
    assert [r.name for r in recipes] == ["Eggs on Toast", "Chicken Bowl"]
    assert recipes[0].steps == ["Boil the eggs.", "Toast the bread."]
    assert recipes[0].portion_size == "1 serving"


def test_step_order_is_kept_for_ten_steps(data_manager):
    steps = [f"Step number {n}." for n in range(1, 11)]
    data_manager.save_profile(make_profile())
    data_manager.save_plan(plan_with(planned(steps=steps)), "nisha")
    loaded = next(iter(data_manager.plans_for_user("nisha").values()))
    assert loaded.meals["breakfast"].recipe_steps == steps


def test_eat_out_meals_store_no_recipe(data_manager):
    data_manager.save_profile(make_profile())
    data_manager.save_plan(plan_with(planned("lunch", source="eat_out", steps=(), ingredients=[],
                                             portion_size="")), "nisha")
    loaded = next(iter(data_manager.plans_for_user("nisha").values()))
    assert loaded.meals["lunch"].recipe_steps == []


def test_json_plans_from_before_the_feature_load_without_recipes(tmp_path):
    old_plan = plan_with(planned()).to_dict()
    for meal_data in old_plan["meals"].values():
        meal_data.pop("recipe_steps"), meal_data.pop("portion_size")
    users, plans = tmp_path / "u.json", tmp_path / "p.json"
    users.write_text(json.dumps({}))
    plans.write_text(json.dumps({"abc": {"user_id": "nisha", **old_plan}}))
    manager = JsonDataManager(str(users), str(plans))
    loaded = manager.plans_for_user("nisha")
    assert loaded and next(iter(loaded.values())).meals["breakfast"].recipe_steps == []


def test_an_existing_sqlite_database_is_upgraded_in_place(tmp_path):
    path = str(tmp_path / "calora.db")
    manager = SqliteDataManager(path)
    manager.save_profile(make_profile())
    manager.save_plan(plan_with(planned()), "nisha")
    import sqlite3
    connection = sqlite3.connect(path)
    connection.executescript("DROP TABLE meal_recipe_steps; DROP TABLE meal_recipes;")  # a database from before
    connection.close()
    upgraded = SqliteDataManager(path)  # recreates the new tables without touching old data
    old = next(iter(upgraded.plans_for_user("nisha").values()))
    assert old.meals["breakfast"].name == "Eggs on Toast" and old.meals["breakfast"].recipe_steps == []
    upgraded.save_plan(plan_with(planned()), "nisha")
    newest = list(upgraded.plans_for_user("nisha").values())[-1]
    assert newest.meals["breakfast"].recipe_steps == ["Boil the eggs.", "Toast the bread."]


# ---- terminal output ---------------------------------------------------------------------
def test_io_formats_recipes_and_prints_nothing_when_there_are_none():
    shown = []
    io = IOManager(output_fn=lambda *a: shown.append(" ".join(map(str, a))))
    recipes = [Recipe("breakfast", "Eggs on Toast", "1 serving", ["2 eggs"], ["Boil.", "Serve."])]
    text = io.format_recipes(recipes)
    assert text.startswith("=== Step-by-step recipes ===") and "1. Boil." in text and "2. Serve." in text
    assert io.format_recipes([]) == ""
    io.show_recipes([])
    assert shown == []
    io.show_recipes(recipes)
    assert "Eggs on Toast" in shown[0]


def test_cli_shows_recipes_after_a_home_cooked_plan(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    out = run_cli(controller, NEW_USER + ["1", "rice dishes", "in", "eggs, rice", "0", "", "5"])
    assert "=== Step-by-step recipes ===" in out
    assert "1. " in out and "Steps:" in out


def test_cli_shows_no_recipes_when_eating_out(controller_factory, response_text):
    controller, _ = controller_factory(response_text)
    out = run_cli(controller, NEW_USER + ["1", "rice dishes", "out", "", "0", "", "5"])
    assert "=== Your Meal Plan ===" in out
    assert "Step-by-step recipes" not in out


# ---- web page -----------------------------------------------------------------------------
@pytest.fixture
def web_with_recipes(controller_factory):
    from app import WebApp
    data = sample_response()
    for item in data["recommendations"]:
        if item["meal_source_type"] == "home_cooked":
            item["recipe_steps"] = ["Chop the <script>alert(1)</script> onions.", "Serve & enjoy."]
    controller, _ = controller_factory(json.dumps(data))
    return WebApp(controller).flask_app.test_client()


def test_web_plan_page_lists_recipes_and_escapes_the_ai_text(web_with_recipes):
    client = web_with_recipes
    sign_up(client)
    job_id = start_plan(client)
    assert wait_for_plan(client, job_id) == "DONE"
    page = client.get(f"/plan/result/{job_id}").get_data(as_text=True)
    assert "<h2>Recipes</h2>" in page and "recipe-steps" in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<script>alert(1)" not in page
    assert "Serve &amp; enjoy." in page


def test_web_history_keeps_the_recipes(web_with_recipes):
    client = web_with_recipes
    sign_up(client)
    job_id = start_plan(client)
    wait_for_plan(client, job_id)
    history = client.get("/plans").get_data(as_text=True)
    assert "recipe-steps" in history and "Serve &amp; enjoy." in history


def test_web_plan_without_home_cooked_meals_has_no_recipe_section(web_with_recipes):
    client = web_with_recipes
    sign_up(client)
    job_id = start_plan(client, {**PLAN_FORM, "meal_source": "out", "ingredients": ""})
    wait_for_plan(client, job_id)
    page = client.get(f"/plan/result/{job_id}").get_data(as_text=True)
    assert "<h2>Recipes</h2>" not in page
