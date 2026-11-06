"""Domain model classes: encapsulation, normalisation and stored shapes."""
import pytest

from manager.models import (
    IngredientParser, MealCandidate, MealPlan, MealPlanRequest, NutritionTargets,
    PlannedMeal, ShoppingList, UserProfile,
)
from tests.conftest import make_profile, meal


def test_ingredient_parser_handles_strings_and_objects():
    raw = ["2 eggs", {"item": "chicken", "quantity": 150, "unit": "g"}, {}, "", None, 5]
    assert IngredientParser.names(raw) == ["2 eggs", "chicken"]
    assert IngredientParser.labels(raw) == ["2 eggs", "150 g chicken"]
    assert IngredientParser.names(None) == []


@pytest.mark.parametrize("raw,expected", [("M", "male"), ("m", "male"), ("MALE", "male"),
                                          ("F", "female"), ("female", "female")])
def test_gender_is_normalised(raw, expected):
    assert make_profile(gender=raw).gender == expected


@pytest.mark.parametrize("raw,expected", [("moderately active", "moderate"), ("very active", "very_active"),
                                          ("Sedentary", "sedentary"), ("light", "light")])
def test_activity_level_is_normalised(raw, expected):
    assert make_profile(activity_level=raw).activity_level == expected


@pytest.mark.parametrize("raw", ["nil", "none", "no", "no restrictions", "", "n/a", None, []])
def test_no_restriction_words_give_empty_list(raw):
    assert make_profile(dietary_restrictions=raw).dietary_restrictions == []


@pytest.mark.parametrize("raw,expected", [
    ("fish allergy", ["fish"]), ("allergy to fish", ["fish"]), ("allergic to fish", ["fish"]),
    ("I am allergic to fish", ["fish"]), ("peanut, shellfish", ["peanut", "shellfish"]),
    ("allergic to peanuts and dairy", ["peanuts", "dairy"]), ("Peanut; Egg", ["peanut", "egg"]),
    (["PEANUT", "  Dairy "], ["peanut", "dairy"])])
def test_restrictions_are_split_into_individual_terms(raw, expected):
    assert UserProfile.normalize_restrictions(raw) == expected


def test_profile_fields_are_read_only():
    profile = make_profile()
    with pytest.raises(AttributeError):
        profile.age = 99
    assert profile.age == 21
    # private by naming convention: the double underscore mangles the attribute name
    assert not hasattr(profile, "__age")
    assert hasattr(profile, "_UserProfile__age")


def test_profile_returns_copies_of_lists():
    profile = make_profile(dietary_restrictions=["peanut"])
    profile.dietary_restrictions.append("egg")
    assert profile.dietary_restrictions == ["peanut"]


def test_profile_round_trip_keeps_preferences():
    profile = make_profile()
    profile.remember_preferences({"meal_preference": "rice", "meal_source": "home_cooked",
                                  "pantry_ingredients": ["egg"], "daily_budget": 20.0, "ignored": 1})
    data = profile.to_dict()
    assert data["meal_preference"] == "rice" and "ignored" not in data
    assert UserProfile.from_dict(data).to_dict() == data
    assert "meal_preference" not in profile.to_dict(include_preferences=False)


def test_part1_cli_record_still_loads():
    """Records saved by the Part 1 CLI used 'M' and a free-text restriction string."""
    legacy = {"username": "sam", "name": "Sam", "age": 30, "gender": "M", "weight_kg": 70.0,
              "height_cm": 175.0, "activity_level": "moderately active", "goal": "lose",
              "dietary_restrictions": "allergic to peanuts"}
    profile = UserProfile.from_dict(legacy)
    assert (profile.gender, profile.activity_level) == ("male", "moderate")
    assert profile.dietary_restrictions == ["peanuts"]


def test_part1_web_record_still_loads():
    legacy = {"username": "tim", "name": "timothy", "age": 23, "gender": "male", "weight_kg": 75.0,
              "height_cm": 180.0, "activity_level": "moderate", "goal": "gain",
              "dietary_restrictions": ["fish"], "meal_preference": "savoury", "meal_source": "both",
              "pantry_ingredients": [], "daily_budget": 40.0}
    assert UserProfile.from_dict(legacy).to_dict() == legacy


def test_updated_with_returns_new_profile_and_leaves_original():
    profile = make_profile()
    updated = profile.updated_with({"age": 30, "goal": None, "dietary_restrictions": ["egg"]})
    assert (updated.age, updated.goal, updated.dietary_restrictions) == (30, "maintain", ["egg"])
    assert profile.age == 21


@pytest.mark.parametrize("source,canonical,label", [
    ("in", "home_cooked", "in"), ("home_cooked", "home_cooked", "in"), ("out", "eat_out", "out"),
    ("eat_out", "eat_out", "out"), ("both", "both", "both"), ("anything", "both", "both"), ("", "both", "both")])
def test_request_normalises_meal_source(source, canonical, label):
    request = MealPlanRequest(make_profile(), meal_source=source)
    assert (request.meal_source, request.meal_source_label) == (canonical, label)


def test_request_refers_to_profile_and_copies_lists():
    profile = make_profile()
    request = MealPlanRequest(profile, "rice", "both", "eggs, rice ,", 20)
    assert request.profile is profile
    assert request.pantry_ingredients == ["eggs", "rice"]
    request.pantry_ingredients.append("x")
    assert request.pantry_ingredients == ["eggs", "rice"]
    assert request.to_preferences()["daily_budget"] == 20


def test_nutrition_targets_expose_values_read_only():
    targets = NutritionTargets(1, 2, 3, 4, {"protein_g": 1}, {"protein_g": 2})
    assert (targets.bmr, targets.tdee, targets.daily_calories, targets.per_meal_calories) == (1, 2, 3, 4)
    targets.daily_macros["protein_g"] = 99
    assert targets.daily_macros == {"protein_g": 1}
    with pytest.raises(AttributeError):
        targets.bmr = 5


def test_meal_candidate_from_dict_and_slot():
    candidate = MealCandidate.from_dict(meal("  BREAKFAST\t", ingredients=[{"item": "rice"}, "egg"]))
    assert candidate.slot == "breakfast"
    assert candidate.ingredient_names == ["rice", "egg"]
    assert candidate.source_type == "home_cooked"
    empty = MealCandidate.from_dict({})
    assert (empty.calories, empty.estimated_cost, empty.ingredients) == (0, None, [])


def _planned(slot, calories, cost, source="home_cooked", ingredients=None):
    return PlannedMeal(slot, slot.title() + " dish", source, "Moderate (400-700 kcal)", "on_target",
                       "High (>20-30g)", "High (>30-60g)", "High (>10-20g)", "Budget (<$5)",
                       ingredients, calories, cost)


def test_plan_owns_meals_and_computes_totals():
    plan = MealPlan(1500, 10)
    plan.add_meal(_planned("breakfast", 400, 3))
    plan.add_meal(_planned("lunch", 500, 4.5))
    plan.mark_missing("dinner")
    assert plan.total_calories == 900 and plan.total_cost == 7.5
    assert plan.is_complete is False and plan.missing_slots == ["dinner"]
    assert plan.within_budget is True
    assert list(plan.meals) == ["breakfast", "lunch"]


@pytest.mark.parametrize("budget,within", [(None, True), (0, False), (2.49, False), (2.5, True)])
def test_budget_boundaries(budget, within):
    plan = MealPlan(1500, budget)
    plan.add_meal(_planned("breakfast", 400, 2.5))
    assert plan.within_budget is within


def test_unknown_and_free_costs():
    plan = MealPlan(1500, 5)
    plan.add_meal(_planned("breakfast", 400, None))
    assert plan.total_cost is None and plan.within_budget is True
    free = MealPlan(1500, 5)
    free.add_meal(_planned("breakfast", 400, 0))
    assert free.total_cost == 0
    mixed = MealPlan(1500, 5)
    mixed.add_meal(_planned("breakfast", 400, None))
    mixed.add_meal(_planned("lunch", 400, 2))
    assert mixed.total_cost == 2


def test_explanation_wording():
    plan = MealPlan(1542.3, 30)
    plan.add_meal(_planned("breakfast", 1400, 11))
    assert plan.explanation == ("Totals 1400 kcal against a 1542 kcal daily target; "
                                "an estimated 11.00 — within your 30.00 daily budget.")
    over = MealPlan(1542.3, 10)
    over.add_meal(_planned("breakfast", 1400, 11))
    assert "over your" in over.explanation
    no_budget = MealPlan(1542.3, None)
    no_budget.add_meal(_planned("breakfast", 1400, 11))
    assert no_budget.explanation == "Totals 1400 kcal against a 1542 kcal daily target."


def test_plan_dict_has_the_part1_shape_and_round_trips():
    plan = MealPlan(1500, 20)
    plan.add_meal(_planned("breakfast", 400, 3, ingredients=["2 eggs"]))
    plan.mark_missing("lunch")
    data = plan.to_dict()
    assert list(data) == ["meals", "is_complete", "missing_slots", "daily_calorie_target", "total_calories",
                          "daily_budget", "total_cost", "within_budget", "explanation"]
    assert data["meals"]["breakfast"]["pantry_ingredients_used"] == ["2 eggs"]
    assert list(data["meals"]["breakfast"]) == ["name", "source_type", "calorie_range", "calorie_tag",
                                                "protein_tag", "carbs_tag", "fat_tag", "cost_tier",
                                                "pantry_ingredients_used"]
    assert MealPlan.from_dict(data).to_dict() == data


def test_part1_stored_plan_loads():
    stored = {"meals": {"lunch": {"name": "Spicy Beef Hor Fun", "source_type": "eat_out",
                                  "calorie_range": "High (>700 kcal)", "calorie_tag": "on_target",
                                  "protein_tag": "Very High (>30g)", "carbs_tag": "Very High (>60g)",
                                  "fat_tag": "Very High (>20g)", "cost_tier": "Moderate (<$10)",
                                  "pantry_ingredients_used": []}},
              "is_complete": False, "missing_slots": ["breakfast", "dinner"], "daily_calorie_target": 3035.75,
              "total_calories": 1015, "daily_budget": 40, "total_cost": 8.0, "within_budget": True,
              "explanation": "x", "user_id": "tim"}
    plan = MealPlan.from_dict(stored)
    assert plan.total_cost == 8.0 and plan.missing_slots == ["breakfast", "dinner"]
    assert plan.meals["lunch"].name == "Spicy Beef Hor Fun"


def test_shopping_list_text():
    assert ShoppingList().is_empty
    assert "Nothing to buy" in ShoppingList().to_text()
    text = ShoppingList(["rice", "egg"], ["salt"]).to_text()
    assert "  - rice" in text and "Already in your pantry: salt" in text
    items = ShoppingList(["rice"]).items
    items.append("x")
    assert ShoppingList(["rice"]).items == ["rice"]
