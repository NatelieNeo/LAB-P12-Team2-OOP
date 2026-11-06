"""LogicManager and the classes it is built from. All offline."""
import pytest

from manager.logic_manager import (
    CalorieFitCriterion, CostCriterion, DietarySafetyRule, LogicManager, MacroFitCriterion,
    MealTagger, NutritionCalculator, PreferenceCriterion, SourcePreferenceRule,
)
from manager.models import MealCandidate, MealPlan
from tests.conftest import eat_out, make_profile, make_request, meal


def cand(**kwargs):
    return MealCandidate.from_dict(meal(**kwargs))


# ---- NutritionCalculator ---------------------------------------------------
@pytest.fixture
def calc():
    return NutritionCalculator()


def test_bmr_uses_mifflin_st_jeor(calc):
    assert calc.bmr(make_profile(gender="male", weight_kg=70, height_cm=175, age=30)) == 1648.75
    assert calc.bmr(make_profile(gender="female", weight_kg=70, height_cm=175, age=30)) == 1482.75


@pytest.mark.parametrize("gender", ["male", "MALE", "M", "m"])
def test_male_aliases_use_the_male_formula(calc, gender):
    assert calc.bmr(make_profile(gender=gender, weight_kg=70, height_cm=175, age=30)) == 1648.75


def test_full_target_pipeline(calc):
    targets = calc.targets(make_profile())
    assert targets.bmr == 1285.25
    assert targets.tdee == pytest.approx(1542.3)
    assert targets.daily_calories == pytest.approx(1542.3)
    assert targets.per_meal_calories == pytest.approx(514.1)
    assert targets.per_meal_macros == {"protein_g": 38.6, "carbs_g": 51.4, "fat_g": 17.1}


@pytest.mark.parametrize("activity,multiplier", [
    ("sedentary", 1.2), ("light", 1.375), ("moderate", 1.55), ("moderately active", 1.55),
    ("active", 1.725), ("very_active", 1.9), ("very active", 1.9), ("unknown", 1.2)])
def test_activity_multipliers(calc, activity, multiplier):
    profile = make_profile(activity_level=activity)
    assert calc.tdee(profile) == pytest.approx(calc.bmr(profile) * multiplier)


@pytest.mark.parametrize("goal,target,macros", [
    ("lose", 1000, {"protein_g": 78.8, "carbs_g": 78.8, "fat_g": 30}),
    ("maintain", 1500, {"protein_g": 67.5, "carbs_g": 90, "fat_g": 30}),
    ("gain", 1800, {"protein_g": 78.8, "carbs_g": 90, "fat_g": 25}),
    ("unknown", 1500, {"protein_g": 67.5, "carbs_g": 90, "fat_g": 30})])
def test_goal_adjustments_and_macro_splits(calc, goal, target, macros):
    assert calc.daily_calorie_target(1500, goal) == target
    assert calc.macro_targets(900, goal) == macros


def test_per_meal_target_splits_across_three_meals(calc):
    assert calc.per_meal_calorie_target(1500) == 500
    assert calc.per_meal_calorie_target(0) == 0


# ---- DietarySafetyRule -------------------------------------------------------
def safe(candidates, restrictions):
    request = make_request(make_profile(dietary_restrictions=restrictions))
    return DietarySafetyRule().apply(candidates, request)


def test_filter_checks_name_and_ingredients():
    items = [cand(name="Peanut Butter Toast", ingredients=[]), cand(ingredients=[{"item": "peanut oil"}]), cand()]
    assert safe(items, ["PEANUT"]) == [items[2]]
    assert safe(items, []) == items
    assert safe(items, ["no peanut"]) == items  # literal text, not a negation
    assert safe(items, [""]) == items


def test_hidden_allergen_in_eat_out_ingredients_is_filtered():
    hidden = cand(slot="Lunch", name="Economical Rice with Chicken", meal_source_type="eat_out",
                  ingredients=["rice", "chicken", "fish sauce"])
    plain = cand(slot="Lunch", name="Chicken Rice", meal_source_type="eat_out", ingredients=["rice", "chicken"])
    assert safe([hidden, plain], "fish allergy") == [plain]


def test_eat_out_dish_with_unknown_ingredients_is_excluded():
    unknown = cand(slot="Lunch", name="Mystery", meal_source_type="eat_out", ingredients=[])
    assert safe([unknown], "fish") == []


@pytest.mark.parametrize("no_restrictions", ["nil", "none", "no", "no restrictions", ""])
def test_textual_no_restrictions_filters_nothing(no_restrictions):
    item = cand()
    assert safe([item], no_restrictions) == [item]


def test_comma_separated_restrictions_are_separate_terms():
    peanut, plain = cand(name="Peanut Butter Toast"), cand(name="Eggs on Toast")
    assert safe([peanut, plain], "peanut, shellfish") == [plain]


@pytest.mark.parametrize("restriction", ["fish allergy", "allergy to fish", "allergic to fish", "I am allergic to fish"])
def test_allergy_phrasing_filters_by_name(restriction):
    fish = cand(slot="Lunch", name="Economical Rice with Chicken and Fish", meal_source_type="eat_out", ingredients=[])
    ok = cand(slot="Lunch", name="Chicken Rice", meal_source_type="eat_out", ingredients=["chicken", "rice"])
    assert safe([fish, ok], restriction) == [ok]


@pytest.mark.parametrize("restriction,ingredients", [
    ("peanut", ["peanut oil", "rice"]), ("peanuts", ["peanut oil", "rice"]), ("nuts", ["almond", "oats"]),
    ("dairy", ["cheese", "bread"]), ("shellfish", ["prawn", "rice"]), ("gluten", ["wheat flour"]),
    ("egg", ["mayonnaise"]), ("soy", ["tofu"]), ("fish", ["anchovy"])])
def test_restriction_categories_expand_to_related_ingredients(restriction, ingredients):
    assert safe([cand(ingredients=ingredients)], restriction) == []


def test_risky_dish_removed_even_if_allergen_not_listed():
    satay = cand(slot="Lunch", name="Chicken Satay", meal_source_type="eat_out", ingredients=["chicken", "sauce"])
    assert safe([satay], "peanut") == []


def test_coconut_is_not_mistaken_for_a_nut():
    item = cand(name="Coconut rice", ingredients=["coconut", "rice"])
    assert safe([item], "nut") == [item]


def test_custom_keyword_table_can_be_injected():
    rule = DietarySafetyRule(keywords={"spicy": ["chilli"]}, risky_dishes={})
    request = make_request(make_profile(dietary_restrictions=["spicy"]))
    assert rule.apply([cand(ingredients=["chilli"])], request) == []


# ---- SourcePreferenceRule ----------------------------------------------------
@pytest.mark.parametrize("source,count", [("home_cooked", 1), ("eat_out", 1), ("both", 2), ("in", 1), ("", 2)])
def test_source_filter(source, count):
    items = [cand(), cand(meal_source_type="eat_out")]
    request = make_request(make_profile(), meal_source=source)
    assert len(SourcePreferenceRule().apply(items, request)) == count


def test_source_filter_is_soft_when_nothing_matches():
    only_out = [cand(meal_source_type="eat_out")]
    request = make_request(make_profile(), meal_source="in")
    assert SourcePreferenceRule().apply(only_out, request) == only_out


# ---- Criteria and ranking ----------------------------------------------------
@pytest.fixture
def targets():
    return NutritionCalculator().targets(make_profile())


def test_individual_criteria_scores(targets):
    request = make_request(make_profile(), meal_preference="western")
    assert CalorieFitCriterion().score(cand(calories=450), targets, request) == pytest.approx(abs(450 - 514.1))
    macros = targets.per_meal_macros
    perfect = cand(protein_g=macros["protein_g"], carbs_g=macros["carbs_g"], fat_g=macros["fat_g"])
    assert MacroFitCriterion().score(perfect, targets, request) == 0
    assert PreferenceCriterion().score(cand(name="Grilled Chicken Sandwich"), targets, request) == 0.0
    assert PreferenceCriterion().score(cand(name="Chicken Rice"), targets, request) == 1.0
    assert CostCriterion().score(cand(cost=0), targets, request) == 0
    assert CostCriterion().score(cand(cost=None), targets, request) == float("inf")


def test_ranking_order_and_inputs_untouched(targets):
    manager = LogicManager()
    request = make_request(make_profile(), meal_preference="")
    per_meal = targets.per_meal_calories
    macros = targets.per_meal_macros
    exact = dict(calories=per_meal, protein_g=macros["protein_g"], carbs_g=macros["carbs_g"], fat_g=macros["fat_g"])
    best = cand(name="Best", cost=5, **exact)
    cheaper = cand(name="Cheaper", cost=0, **exact)
    worse_macro = cand(name="Less Protein", cost=0, **{**exact, "protein_g": macros["protein_g"] - 5})
    worse_calorie = cand(name="Low Calories", cost=0, **{**exact, "calories": per_meal - 1})
    items = [worse_calorie, worse_macro, best, cheaper]
    ranked = manager.rank_slot(items, targets, request)
    assert [c.meal_name for c in ranked] == ["Cheaper", "Best", "Less Protein", "Low Calories"]
    assert [c.meal_name for c in items] == ["Low Calories", "Less Protein", "Best", "Cheaper"]
    assert manager.rank_slot([], targets, request) == []


def test_cuisine_breaks_a_tie_after_nutrition_fit(targets):
    manager = LogicManager()
    request = make_request(make_profile(), meal_preference="western")
    exact = dict(calories=targets.per_meal_calories, protein_g=25, carbs_g=40, fat_g=15, cost=5)
    chinese = cand(slot="Lunch", name="Chicken Rice", **exact)
    western = cand(slot="Lunch", name="Grilled Chicken Sandwich", **exact)
    assert manager.rank_slot([chinese, western], targets, request)[0].meal_name == "Grilled Chicken Sandwich"


def test_cuisine_does_not_override_better_nutrition_fit(targets):
    manager = LogicManager()
    request = make_request(make_profile(), meal_preference="western")
    fits = dict(calories=targets.per_meal_calories, protein_g=25, carbs_g=40, fat_g=15, cost=5)
    sandwich = cand(slot="Lunch", name="Grilled Chicken Sandwich", **{**fits, "calories": 300})
    rice = cand(slot="Lunch", name="Chicken Rice", **fits)
    assert manager.rank_slot([sandwich, rice], targets, request)[0].meal_name == "Chicken Rice"


def test_a_wrong_source_dish_survives_only_when_nothing_else_remains(targets):
    manager = LogicManager()
    request = make_request(make_profile(), meal_source="in")
    out, home = cand(meal_source_type="eat_out"), cand(name="Home")
    assert manager.rank_slot([out, home], targets, request)[0].meal_name == "Home"
    assert manager.rank_slot([out], targets, request)[0].source_type == "eat_out"


# ---- MealTagger --------------------------------------------------------------
@pytest.mark.parametrize("method,cases", [
    ("tag_calorie_range", [(399.99, "Low (<400 kcal)"), (400, "Moderate (400-700 kcal)"),
                           (700, "Moderate (400-700 kcal)"), (700.01, "High (>700 kcal)")]),
    ("tag_protein", [(9.99, "Low (<10g)"), (10, "Moderate (10-20g)"), (20, "Moderate (10-20g)"),
                     (20.01, "High (>20-30g)"), (30, "High (>20-30g)"), (30.01, "Very High (>30g)")]),
    ("tag_carbs", [(14.99, "Low (<15g)"), (15, "Moderate (15-30g)"), (30, "Moderate (15-30g)"),
                   (30.01, "High (>30-60g)"), (60, "High (>30-60g)"), (60.01, "Very High (>60g)")]),
    ("tag_fat", [(4.99, "Low (<5g)"), (5, "Moderate (5-10g)"), (10, "Moderate (5-10g)"),
                 (10.01, "High (>10-20g)"), (20, "High (>10-20g)"), (20.01, "Very High (>20g)")]),
    ("tag_cost", [(None, "Unknown"), (0, "Budget (<$5)"), (4.99, "Budget (<$5)"), (5, "Moderate (<$10)"),
                  (9.99, "Moderate (<$10)"), (10, "Moderate (<$15)"), (14.99, "Moderate (<$15)"),
                  (15, "High (<$20)"), (19.99, "High (<$20)"), (20, "Premium (>$20)")]),
])
def test_tag_boundaries(method, cases):
    tagger = MealTagger()
    for value, expected in cases:
        assert getattr(tagger, method)(value) == expected


@pytest.mark.parametrize("calories,expected", [(399.99, "under_target"), (400, "on_target"),
                                               (600, "on_target"), (600.01, "over_target")])
def test_calorie_fit_has_an_inclusive_twenty_percent_band(calories, expected):
    assert MealTagger().tag_calorie_fit(calories, 500) == expected
    assert MealTagger(tolerance=0).tag_calorie_fit(500, 500) == "on_target"


def test_annotate_produces_tags_not_numbers(targets):
    planned = MealTagger().annotate(cand(calories=514.1), targets)
    assert planned.to_dict() == {
        "name": "Eggs on Toast", "source_type": "home_cooked", "calorie_range": "Moderate (400-700 kcal)",
        "calorie_tag": "on_target", "protein_tag": "High (>20-30g)", "carbs_tag": "High (>30-60g)",
        "fat_tag": "High (>10-20g)", "cost_tier": "Budget (<$5)",
        "pantry_ingredients_used": ["2 eggs", "2 slices wholemeal bread"]}
    assert planned.calories == 514.1


def test_eat_out_meals_carry_no_ingredients(targets):
    planned = MealTagger().annotate(cand(meal_source_type="eat_out", ingredients=["rice"]), targets)
    assert planned.ingredients == []


# ---- LogicManager.build_meal_plan ---------------------------------------------
def test_happy_path_plan(candidates, request_both):
    plan = LogicManager().build_meal_plan(candidates, request_both)
    assert plan.is_complete and plan.missing_slots == []
    assert {slot: meal_.name for slot, meal_ in plan.meals.items()} == {
        "breakfast": "Eggs on Toast", "lunch": "Chicken Rice", "dinner": "Chicken and Rice Bowl"}
    assert plan.daily_calorie_target == pytest.approx(1542.3)
    assert plan.total_calories == 1420
    assert plan.total_cost == 10.5 and plan.within_budget is True
    assert plan.meals["dinner"].ingredients == ["150g chicken breast", "1 cup cooked rice"]


def test_group_by_slot_drops_unknown_meal_types():
    manager = LogicManager()
    items = [cand(slot="  BREAKFAST\t"), cand(slot="Lunch"), cand(slot="Snack"), MealCandidate.from_dict({})]
    grouped = manager.group_by_slot(items)
    assert [len(grouped[s]) for s in MealPlan.SLOTS] == [1, 1, 0]
    assert manager.group_by_slot(None) == {slot: [] for slot in MealPlan.SLOTS}


def test_empty_and_fully_filtered_plans_return_none(profile):
    manager = LogicManager()
    request = make_request(profile)
    assert manager.build_meal_plan([], request) is None
    assert manager.build_meal_plan(None, request) is None
    assert manager.build_meal_plan([cand(slot="Snack")], request) is None
    peanut = make_request(make_profile(dietary_restrictions=["peanut"]))
    assert manager.build_meal_plan([cand(name="Peanut Toast")], peanut) is None


def test_partial_plan_reports_missing_slots(request_both):
    plan = LogicManager().build_meal_plan([cand()], request_both)
    assert plan.is_complete is False
    assert plan.missing_slots == ["lunch", "dinner"]
    assert list(plan.meals) == ["breakfast"]


def test_allergy_phrasing_drives_the_whole_plan(response):
    items = [cand(slot="Breakfast"), cand(slot="Lunch", name="Economical Rice with Chicken and Fish",
             meal_source_type="eat_out", ingredients=[]),
             cand(slot="Lunch", name="Chicken Rice", meal_source_type="eat_out", ingredients=["chicken", "rice"]),
             cand(slot="Dinner")]
    request = make_request(make_profile(dietary_restrictions="I am allergic to fish"))
    assert LogicManager().build_meal_plan(items, request).meals["lunch"].name == "Chicken Rice"


@pytest.mark.parametrize("budget,within", [(None, True), (2.49, False), (2.5, True)])
def test_plan_budget_boundaries(profile, budget, within):
    request = make_request(profile, daily_budget=budget)
    assert LogicManager().build_meal_plan([cand()], request).within_budget is within


def test_inputs_are_not_modified(candidates, request_both):
    before = [c.meal_name for c in candidates]
    LogicManager().build_meal_plan(candidates, request_both)
    assert [c.meal_name for c in candidates] == before


def test_bad_numeric_data_surfaces_as_an_error(profile):
    with pytest.raises(TypeError):
        LogicManager().build_meal_plan([cand(calories="400")], make_request(profile))
