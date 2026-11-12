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



