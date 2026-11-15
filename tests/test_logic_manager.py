"""Unit tests for the Logic Manager, run entirely on hard-coded sample AI responses.

No live AI API is used anywhere in this file. The AI's output is represented by
fixed JSON text or fixed dictionaries (see ``tests/helpers.py``), parsed by the
real AIHandlerManager, and then handed to the real LogicManager classes:

* NutritionCalculator  - BMR -> TDEE -> goal -> per-meal calorie/macro targets
* DietarySafetyRule / SourcePreferenceRule (MealRule interface, polymorphic)
* Calorie/Macro/Preference/Cost criteria (RankingCriterion interface)
* MealTagger, ShoppingListBuilder, RecipeBuilder (the new features)
* LogicManager: grouping, ranking, plan building, strict budget, and the
  scoring/selection of complete AI response sets.
"""
import json

import pytest

import config
from manager.ai_handler_manager import AIHandlerManager
from manager.exceptions import AIResponseError, NoSuitableMealsError
from manager.logic_manager import (
    CalorieFitCriterion, CostCriterion, DietarySafetyRule, LogicManager, MacroFitCriterion,
    MealRule, MealTagger, NutritionCalculator, PreferenceCriterion, RankingCriterion,
    RecipeBuilder, ShoppingListBuilder, SourcePreferenceRule,
)
from manager.models import AIResponseSet, MealCandidate, MealPlan
from tests.helpers import (
    eat_out, make_profile, make_request, meal, response_data, sample_response,
)

PER_MEAL = 1285.25 * 1.2 / 3  # default test profile: 21 y, female, 52 kg, 165 cm, sedentary, maintain


# ---- small builders ---------------------------------------------------------------------------
def cand(slot="breakfast", name="Eggs on Toast", calories=500, cost=3.0, source="home_cooked", **changes):
    return MealCandidate.from_dict(meal(slot, name, calories, cost, source, **changes))


def response_set(provider, data):
    candidates = tuple(MealCandidate.from_dict(item) for item in data["recommendations"])
    return AIResponseSet(provider, candidates, 0.0, 1.0)


def planned(slot, name, ingredients, source="home_cooked", steps=None):
    """A PlannedMeal produced by the real MealTagger from a candidate."""
    extra = {"ingredients": ingredients}
    if steps is not None:
        extra["recipe_steps"] = steps
    candidate = cand(slot, name, 500, 3.0, source, **extra)
    targets = NutritionCalculator().targets(make_profile())
    return MealTagger().annotate(candidate, targets)


def plan_of(*meals):
    plan = MealPlan(1500)
    for item in meals:
        plan.add_meal(item)
    return plan


def day(cheap, dear, calories_cheap=300, calories_dear=514):
    """Two options per slot: a cheap one far from the calorie target and a dear one on target."""
    out = []
    for slot in MealPlan.SLOTS:
        out.append(cand(slot, f"{slot} cheap", calories_cheap, cheap))
        out.append(cand(slot, f"{slot} dear", calories_dear, dear))
    return out


# =============================================================================================
# NutritionCalculator
# =============================================================================================
class TestNutritionCalculator:
    calc = NutritionCalculator()

    def test_bmr_female_and_male_use_mifflin_st_jeor(self):
        assert self.calc.bmr(make_profile()) == pytest.approx(1285.25)
        assert self.calc.bmr(make_profile(gender="male")) == pytest.approx(1451.25)

    @pytest.mark.parametrize("level,factor", [
        ("sedentary", 1.2), ("light", 1.375), ("moderate", 1.55), ("active", 1.725), ("very_active", 1.9)])
    def test_tdee_applies_the_activity_multiplier(self, level, factor):
        profile = make_profile(activity_level=level)
        assert self.calc.tdee(profile) == pytest.approx(self.calc.bmr(profile) * factor)

    def test_unknown_activity_level_falls_back_to_sedentary(self):
        assert self.calc.tdee(make_profile(activity_level="couch")) == pytest.approx(1285.25 * 1.2)

    @pytest.mark.parametrize("goal,adjustment", [("lose", -500), ("maintain", 0), ("gain", 300), ("???", 0)])
    def test_goal_adjusts_the_daily_target(self, goal, adjustment):
        assert self.calc.daily_calorie_target(2000, goal) == 2000 + adjustment

    def test_per_meal_target_is_a_third_of_the_day(self):
        assert self.calc.per_meal_calorie_target(1800) == pytest.approx(600)

    @pytest.mark.parametrize("goal", ["lose", "maintain", "gain"])
    def test_macro_grams_add_back_to_the_calories(self, goal):
        macros = self.calc.macro_targets(600, goal)
        kcal = macros["protein_g"] * 4 + macros["carbs_g"] * 4 + macros["fat_g"] * 9
        assert kcal == pytest.approx(600, abs=2)

    def test_gain_goal_is_more_protein_heavy_than_maintain(self):
        assert (self.calc.macro_targets(600, "gain")["protein_g"]
                > self.calc.macro_targets(600, "maintain")["protein_g"])

    def test_targets_bundles_everything_consistently(self):
        targets = self.calc.targets(make_profile(goal="lose"))
        assert targets.daily_calories == pytest.approx(1285.25 * 1.2 - 500)
        assert targets.per_meal_calories == pytest.approx(targets.daily_calories / 3)
        assert targets.per_meal_macros == self.calc.macro_targets(targets.per_meal_calories, "lose")
        assert targets.daily_macros == self.calc.macro_targets(targets.daily_calories, "lose")


# =============================================================================================
# Business rules (MealRule interface)
# =============================================================================================
class TestDietarySafetyRule:
    rule = DietarySafetyRule()

    def apply(self, candidates, **profile):
        return self.rule.apply(candidates, make_request(make_profile(**profile)))

    def test_it_is_a_meal_rule(self):
        assert isinstance(self.rule, MealRule)

    def test_no_restrictions_keeps_every_candidate(self):
        items = [cand(name="A"), cand(name="B")]
        assert self.apply(items) == items

    def test_ingredient_match_removes_the_dish(self):
        safe, unsafe = cand(name="Plain Rice", ingredients=["rice"]), cand(name="Toast", ingredients=["peanut butter"])
        assert self.apply([safe, unsafe], dietary_restrictions=["peanut"]) == [safe]

    def test_matching_is_case_insensitive(self):
        assert self.apply([cand(ingredients=["Crushed PEANUT"])], dietary_restrictions=["peanut"]) == []

    def test_plural_restriction_also_catches_the_singular_form(self):
        assert self.apply([cand(ingredients=["peanut oil"])], dietary_restrictions=["peanuts"]) == []

    def test_free_text_restriction_is_understood(self):
        items = [cand(name="Milk Oats", ingredients=["milk", "oats"]), cand(name="Plain Rice", ingredients=["rice"])]
        kept = self.apply(items, dietary_restrictions="allergic to peanuts and dairy")
        assert [c.meal_name for c in kept] == ["Plain Rice"]

    def test_a_risky_dish_name_is_removed_even_if_ingredients_look_safe(self):
        satay = cand("lunch", "Chicken Satay", ingredients=["chicken", "skewer"])
        assert self.apply([satay], dietary_restrictions=["peanut"]) == []

    def test_related_words_are_expanded(self):
        # "nuts" bans cashew, almond, ... without the user listing each one
        assert self.apply([cand(ingredients=["cashew"])], dietary_restrictions=["nuts"]) == []

    @pytest.mark.parametrize("ingredient", ["coconut", "butternut squash", "nutmeg"])
    def test_known_safe_words_are_not_mistaken_for_nuts(self, ingredient):
        item = cand(ingredients=[ingredient])
        assert self.apply([item], dietary_restrictions=["nuts"]) == [item]

    def test_vegetarian_removes_meat(self):
        assert self.apply([cand(ingredients=["chicken"])], dietary_restrictions=["vegetarian"]) == []

    def test_unknown_restriction_is_still_matched_as_a_word(self):
        assert self.apply([cand(ingredients=["kiwi"])], dietary_restrictions=["kiwi"]) == []

    def test_multi_field_rule_eat_out_without_ingredients_is_unsafe(self):
        """Source type + ingredient list together decide: safety of an eat-out dish is unknown."""
        unknown = cand("lunch", "Mystery Stall Dish", source="eat_out", ingredients=[])
        known = cand("lunch", "Plain Chicken Rice", source="eat_out", ingredients=["rice", "chicken"])
        assert self.apply([unknown, known], dietary_restrictions=["peanut"]) == [known]

    def test_empty_ingredient_list_is_fine_when_there_is_nothing_to_avoid(self):
        unknown = cand("lunch", "Mystery Stall Dish", source="eat_out", ingredients=[])
        assert self.apply([unknown]) == [unknown]

    def test_input_list_is_not_modified(self):
        items = [cand(ingredients=["peanut"]), cand(name="B", ingredients=["rice"])]
        self.apply(items, dietary_restrictions=["peanut"])
        assert len(items) == 2


class TestSourcePreferenceRule:
    rule = SourcePreferenceRule()
    home, out = cand(name="Home"), cand("breakfast", "Out", source="eat_out")

    def apply(self, source, items):
        return self.rule.apply(items, make_request(make_profile(), meal_source=source))

    def test_it_is_a_meal_rule(self):
        assert isinstance(self.rule, MealRule)

    def test_both_keeps_everything(self):
        assert self.apply("both", [self.home, self.out]) == [self.home, self.out]

    @pytest.mark.parametrize("source,expected", [("in", "Home"), ("home_cooked", "Home"), ("out", "Out")])
    def test_a_source_filters_to_that_source(self, source, expected):
        assert [c.meal_name for c in self.apply(source, [self.home, self.out])] == [expected]

    def test_it_is_a_soft_preference_when_nothing_matches(self):
        assert self.apply("out", [self.home]) == [self.home]


# =============================================================================================
# Ranking criteria (RankingCriterion interface)
# =============================================================================================
class TestRankingCriteria:
    targets = NutritionCalculator().targets(make_profile())
    request = make_request(make_profile())

    def score(self, criterion, candidate, request=None):
        return criterion.score(candidate, self.targets, request or self.request)

    def test_all_criteria_implement_the_interface(self):
        for criterion in (CalorieFitCriterion(), MacroFitCriterion(), PreferenceCriterion(), CostCriterion()):
            assert isinstance(criterion, RankingCriterion)

    def test_calorie_fit_is_distance_from_the_per_meal_target(self):
        assert self.score(CalorieFitCriterion(), cand(calories=PER_MEAL + 40)) == pytest.approx(40)
        assert self.score(CalorieFitCriterion(), cand(calories=PER_MEAL - 40)) == pytest.approx(40)

    def test_macro_fit_sums_the_gram_differences(self):
        macros = self.targets.per_meal_macros
        item = cand(protein_g=macros["protein_g"] + 5, carbs_g=macros["carbs_g"] - 3, fat_g=macros["fat_g"] + 2)
        assert self.score(MacroFitCriterion(), item) == pytest.approx(10)

    def test_cost_is_the_price_and_unknown_cost_ranks_last(self):
        assert self.score(CostCriterion(), cand(cost=4.5)) == 4.5
        assert self.score(CostCriterion(), cand(cost=None)) == float("inf")

    def test_preference_matches_a_craving_word(self):
        crit = PreferenceCriterion()
        assert self.score(crit, cand(name="Chicken Rice")) == 0.0          # "rice dishes" -> rice
        assert self.score(crit, cand(name="Pasta", ingredients=["pasta"])) == 1.0

    def test_preference_matches_a_cuisine(self):
        request = make_request(make_profile(), meal_preference="Japanese food")
        assert self.score(PreferenceCriterion(), cand(name="Salmon Sushi"), request) == 0.0
        assert self.score(PreferenceCriterion(), cand(name="Nasi Lemak"), request) == 1.0

    def test_no_preference_means_every_dish_matches(self):
        request = make_request(make_profile(), meal_preference="")
        assert self.score(PreferenceCriterion(), cand(name="Anything"), request) == 0.0

    def test_filler_words_alone_do_not_exclude_dishes(self):
        request = make_request(make_profile(), meal_preference="food dishes")
        assert self.score(PreferenceCriterion(), cand(name="Anything"), request) == 0.0


# =============================================================================================
# MealTagger
# =============================================================================================
class TestMealTagger:
    tagger = MealTagger()

    @pytest.mark.parametrize("kcal,tag", [
        (399, "Low (<400 kcal)"), (400, "Moderate (400-700 kcal)"),
        (700, "Moderate (400-700 kcal)"), (701, "High (>700 kcal)")])
    def test_calorie_range_boundaries(self, kcal, tag):
        assert self.tagger.tag_calorie_range(kcal) == tag

    @pytest.mark.parametrize("kcal,tag", [
        (399, "under_target"), (400, "on_target"), (600, "on_target"), (601, "over_target")])
    def test_calorie_fit_uses_a_20_percent_band(self, kcal, tag):
        assert self.tagger.tag_calorie_fit(kcal, 500) == tag

    def test_the_tolerance_can_be_injected(self):
        assert MealTagger(tolerance=0.5).tag_calorie_fit(700, 500) == "on_target"

    @pytest.mark.parametrize("grams,tag", [
        (9, "Low (<10g)"), (10, "Moderate (10-20g)"), (20, "Moderate (10-20g)"),
        (30, "High (>20-30g)"), (31, "Very High (>30g)")])
    def test_protein_tags(self, grams, tag):
        assert self.tagger.tag_protein(grams) == tag

    @pytest.mark.parametrize("grams,tag", [
        (14, "Low (<15g)"), (15, "Moderate (15-30g)"), (60, "High (>30-60g)"), (61, "Very High (>60g)")])
    def test_carb_tags(self, grams, tag):
        assert self.tagger.tag_carbs(grams) == tag

    @pytest.mark.parametrize("grams,tag", [
        (4, "Low (<5g)"), (5, "Moderate (5-10g)"), (20, "High (>10-20g)"), (21, "Very High (>20g)")])
    def test_fat_tags(self, grams, tag):
        assert self.tagger.tag_fat(grams) == tag

    @pytest.mark.parametrize("cost,tag", [
        (4.99, "Budget (<$5)"), (5, "Moderate (<$10)"), (12, "Moderate (<$15)"),
        (19.99, "High (<$20)"), (20, "Premium (>$20)"), (None, "Unknown")])
    def test_cost_tiers(self, cost, tag):
        assert self.tagger.tag_cost(cost) == tag

    def test_annotate_a_home_cooked_candidate_keeps_its_recipe(self):
        meal_ = planned("lunch", "Egg Rice", ["rice", "2 eggs"], steps=["Cook rice.", "Fry eggs."])
        assert (meal_.slot, meal_.name, meal_.source_type) == ("lunch", "Egg Rice", "home_cooked")
        assert meal_.ingredients == ["rice", "2 eggs"]
        assert meal_.recipe_steps == ["Cook rice.", "Fry eggs."]
        assert meal_.portion_size == "1 serving"
        assert meal_.calorie_tag == "on_target"

    def test_annotate_an_eat_out_candidate_has_no_recipe_or_portion(self):
        meal_ = planned("dinner", "Chicken Rice", ["rice", "chicken"], source="eat_out")
        assert meal_.ingredients == [] and meal_.recipe_steps == [] and meal_.portion_size == ""


# =============================================================================================
# New feature: shopping list
# =============================================================================================
class TestShoppingListBuilder:
    builder = ShoppingListBuilder()

    def build(self, plan, pantry):
        return self.builder.build(plan, pantry)

    def test_pantry_items_are_split_from_items_to_buy(self):
        plan = plan_of(planned("breakfast", "Eggs on Toast", ["2 eggs", "2 slices wholemeal bread"]))
        result = self.build(plan, ["eggs"])
        assert result.covered_by_pantry == ["2 eggs"]
        assert result.items == ["2 slices wholemeal bread"]

    def test_an_empty_pantry_means_buy_everything(self):
        plan = plan_of(planned("breakfast", "Eggs on Toast", ["2 eggs", "bread"]))
        assert self.build(plan, []).items == ["2 eggs", "bread"]

    def test_everything_covered_gives_an_empty_list(self):
        plan = plan_of(planned("breakfast", "Rice", ["rice"]))
        result = self.build(plan, ["Rice"])
        assert result.is_empty and "Nothing to buy" in result.to_text()

    def test_comma_separated_pantry_entries_are_split(self):
        plan = plan_of(planned("breakfast", "Egg Rice", ["rice", "egg"]))
        assert self.build(plan, ["rice, egg"]).is_empty

    def test_plurals_and_preparation_words_still_match(self):
        plan = plan_of(planned("lunch", "Tomato Egg", ["2 tomato", "chopped onion"]))
        assert self.build(plan, ["tomatoes", "onions"]).is_empty

    def test_a_qualifier_word_still_matches_the_plain_ingredient(self):
        plan = plan_of(planned("breakfast", "Toast", ["2 slices wholemeal bread"]))
        assert self.build(plan, ["bread"]).is_empty

    def test_rice_in_the_pantry_does_not_cover_rice_flour(self):
        plan = plan_of(planned("lunch", "Kueh", ["rice flour"]))
        assert self.build(plan, ["rice"]).items == ["rice flour"]

    def test_eat_out_meals_need_no_shopping(self):
        plan = plan_of(planned("lunch", "Chicken Rice", ["rice", "chicken"], source="eat_out"))
        assert self.build(plan, []).is_empty

    def test_the_same_ingredient_in_two_meals_is_listed_once(self):
        plan = plan_of(planned("breakfast", "A", ["2 eggs"]), planned("lunch", "B", ["2 eggs"]))
        assert self.build(plan, []).items == ["2 eggs"]


# =============================================================================================
# New feature: recipes
# =============================================================================================
class TestRecipeBuilder:
    builder = RecipeBuilder()

    def test_recipes_are_in_breakfast_lunch_dinner_order(self):
        plan = plan_of(planned("dinner", "D", ["x"], steps=["d1"]), planned("breakfast", "B", ["x"], steps=["b1"]),
                       planned("lunch", "L", ["x"], steps=["l1"]))
        assert [r.slot for r in self.builder.build(plan)] == ["breakfast", "lunch", "dinner"]

    def test_eat_out_meals_have_no_recipe(self):
        plan = plan_of(planned("breakfast", "B", ["x"], steps=["b1"]),
                       planned("lunch", "Chicken Rice", ["rice"], source="eat_out"))
        assert [r.name for r in self.builder.build(plan)] == ["B"]

    def test_steps_ingredients_and_portion_are_carried_over_in_order(self):
        plan = plan_of(planned("lunch", "Egg Rice", ["rice", "2 eggs"], steps=["Cook.", "Fry.", "Serve."]))
        recipe = self.builder.build(plan)[0]
        assert recipe.steps == ["Cook.", "Fry.", "Serve."]
        assert recipe.ingredients == ["rice", "2 eggs"]
        assert recipe.portion_size == "1 serving" and recipe.has_steps

    def test_a_plan_with_no_home_cooked_meals_has_no_recipes(self):
        plan = plan_of(planned("lunch", "Chicken Rice", ["rice"], source="eat_out"))
        assert self.builder.build(plan) == []


# =============================================================================================
# LogicManager: grouping, ranking and plan building
# =============================================================================================
class TestGroupingAndRanking:
    manager = LogicManager()
    targets = NutritionCalculator().targets(make_profile())

    def test_group_by_slot_normalises_case_and_whitespace(self):
        grouped = self.manager.group_by_slot([cand(meal_type="Lunch "), cand(meal_type=" DINNER"), cand()])
        assert [len(grouped[s]) for s in MealPlan.SLOTS] == [1, 1, 1]

    def test_group_by_slot_drops_unknown_meal_types(self):
        grouped = self.manager.group_by_slot([cand(meal_type="snack")])
        assert all(items == [] for items in grouped.values())

    def test_group_by_slot_accepts_none(self):
        assert self.manager.group_by_slot(None) == {"breakfast": [], "lunch": [], "dinner": []}

    def test_closest_calories_rank_first(self):
        items = [cand(name="Far", calories=300), cand(name="Close", calories=500), cand(name="Mid", calories=600)]
        ranked = self.manager.rank_slot(items, self.targets, make_request(make_profile()))
        assert [c.meal_name for c in ranked] == ["Close", "Mid", "Far"]

    def test_preference_outranks_cost_when_calories_and_macros_tie(self):
        items = [cand(name="Plain Toast", calories=450, cost=2.0), cand(name="Rice Toast", calories=450, cost=4.0)]
        ranked = self.manager.rank_slot(items, self.targets, make_request(make_profile()))
        assert ranked[0].meal_name == "Rice Toast"

    def test_cost_decides_when_everything_else_ties(self):
        items = [cand(name="Rice A", calories=450, cost=4.0), cand(name="Rice B", calories=450, cost=2.0)]
        assert self.manager.rank_slot(items, self.targets, make_request(make_profile()))[0].meal_name == "Rice B"

    def test_unsafe_dishes_are_removed_before_ranking(self):
        items = [cand(name="Peanut Toast", calories=514, ingredients=["peanut"]),
                 cand(name="Plain Rice", calories=300, ingredients=["rice"])]
        request = make_request(make_profile(dietary_restrictions=["peanut"]))
        assert [c.meal_name for c in self.manager.rank_slot(items, self.targets, request)] == ["Plain Rice"]

    def test_ranking_does_not_modify_the_input_list(self):
        items = [cand(name="Far", calories=300), cand(name="Close", calories=500)]
        self.manager.rank_slot(items, self.targets, make_request(make_profile()))
        assert [c.meal_name for c in items] == ["Far", "Close"]

    # -- Open/Closed and Liskov: new behaviour comes from new classes, not edits to LogicManager --
    def test_ocp_a_different_criteria_list_changes_the_ranking(self):
        items = [cand(name="Close but dear", calories=514, cost=9), cand(name="Far but cheap", calories=300, cost=1)]
        cheapest_first = LogicManager(criteria=[CostCriterion()])
        assert cheapest_first.rank_slot(items, self.targets, make_request(make_profile()))[0].meal_name == "Far but cheap"

    def test_ocp_and_lsp_a_new_rule_subclass_is_used_without_changing_logic_manager(self):
        class NoEggsRule(MealRule):
            def apply(self, candidates, request):
                return [c for c in candidates if not any("egg" in n for n in c.ingredient_names)]

        items = [cand(name="Eggs", ingredients=["eggs"]), cand(name="Tofu", ingredients=["tofu"])]
        ranked = LogicManager(rules=[NoEggsRule()]).rank_slot(items, self.targets, make_request(make_profile()))
        assert [c.meal_name for c in ranked] == ["Tofu"]

    def test_lsp_any_ranking_criterion_subclass_can_be_substituted(self):
        class ShortestName(RankingCriterion):
            def score(self, candidate, targets, request):
                return len(candidate.meal_name)

        items = [cand(name="Long Long Name"), cand(name="Abc")]
        ranked = LogicManager(criteria=[ShortestName()]).rank_slot(items, self.targets, make_request(make_profile()))
        assert ranked[0].meal_name == "Abc"


class TestBuildMealPlan:
    manager = LogicManager()

    def test_a_full_day_picks_one_dish_per_slot_on_target(self):
        items = [cand(s, f"{s} on target", 514) for s in MealPlan.SLOTS] + [cand(s, f"{s} far", 250) for s in MealPlan.SLOTS]
        plan = self.manager.build_meal_plan(items, make_request(make_profile()))
        assert plan.is_complete and plan.missing_slots == []
        assert [m.name for m in plan.meals.values()] == ["breakfast on target", "lunch on target", "dinner on target"]
        assert plan.daily_calorie_target == pytest.approx(PER_MEAL * 3)
        assert plan.total_calories == 514 * 3

    def test_a_slot_with_no_candidates_is_reported_missing(self):
        items = [cand("breakfast"), cand("lunch", "Lunch")]
        plan = self.manager.build_meal_plan(items, make_request(make_profile()))
        assert plan.missing_slots == ["dinner"] and not plan.is_complete
        assert set(plan.meals) == {"breakfast", "lunch"}

    @pytest.mark.parametrize("candidates", [[], None])
    def test_no_candidates_gives_no_plan(self, candidates):
        assert self.manager.build_meal_plan(candidates, make_request(make_profile())) is None

    def test_everything_unsafe_gives_no_plan(self):
        items = [cand(s, f"{s} peanut", ingredients=["peanut"]) for s in MealPlan.SLOTS]
        request = make_request(make_profile(dietary_restrictions=["peanut"]))
        assert self.manager.build_meal_plan(items, request) is None

    def test_one_unsafe_slot_leaves_only_that_slot_missing(self):
        items = [cand("breakfast"), cand("lunch", "L"), cand("dinner", "Peanut Dinner", ingredients=["peanut"])]
        plan = self.manager.build_meal_plan(items, make_request(make_profile(dietary_restrictions=["peanut"])))
        assert plan.missing_slots == ["dinner"]

    def test_meals_carry_tags_computed_by_our_code(self):
        plan = self.manager.build_meal_plan([cand(s) for s in MealPlan.SLOTS], make_request(make_profile()))
        meal_ = plan.meals["lunch"]
        assert meal_.calorie_tag == "on_target" and meal_.cost_tier.startswith("Budget")

    def test_a_soft_budget_still_returns_the_best_plan_but_flags_it(self):
        request = make_request(make_profile(), daily_budget=7)
        plan = self.manager.build_meal_plan(day(cheap=2, dear=6), request)
        assert plan.total_cost == 18.0 and not plan.within_budget
        assert "over" in plan.explanation

    def test_a_strict_budget_chooses_the_cheapest_combination_that_fits(self):
        request = make_request(make_profile(), daily_budget=7, strict_budget=True)
        plan = self.manager.build_meal_plan(day(cheap=2, dear=6), request)
        assert plan.total_cost == 6.0 and plan.within_budget
        assert all(name.endswith("cheap") for name in (m.name for m in plan.meals.values()))

    def test_a_strict_budget_nothing_can_meet_gives_no_plan(self):
        request = make_request(make_profile(), daily_budget=1, strict_budget=True)
        assert self.manager.build_meal_plan(day(cheap=2, dear=6), request) is None

    def test_a_strict_budget_never_accepts_an_unknown_cost(self):
        items = [cand(s, f"{s} ?", cost=None) for s in MealPlan.SLOTS]
        request = make_request(make_profile(), daily_budget=50, strict_budget=True)
        assert self.manager.build_meal_plan(items, request) is None

    def test_shopping_list_and_recipes_come_from_the_plan(self):
        plan = self.manager.build_meal_plan([cand(s, f"{s} meal") for s in MealPlan.SLOTS], make_request(make_profile()))
        shopping = self.manager.build_shopping_list(plan, ["eggs"])
        assert "2 slices wholemeal bread" in shopping.items and "2 eggs" in shopping.covered_by_pantry
        assert [r.slot for r in self.manager.build_recipes(plan)] == ["breakfast", "lunch", "dinner"]


# =============================================================================================
# LogicManager: scoring and choosing a complete AI response (hard-coded AI responses)
# =============================================================================================
class TestScoringAiResponses:
    manager = LogicManager()
    request = make_request(make_profile())

    def test_a_valid_sample_response_gets_a_bounded_score_and_a_complete_plan(self):
        result = self.manager.score_response(response_set("Gemini", sample_response()), self.request)
        assert 0 < result.score <= 100
        assert result.plan.is_complete
        assert set(result.components) == set(config.RESPONSE_SCORE_WEIGHTS)
        assert all(0.0 <= value <= 1.0 for value in result.components.values())

    def test_a_response_that_exactly_meets_the_targets_scores_100(self):
        result = self.manager.score_response(response_set("Gemini", response_data(self.request)), self.request)
        assert result.score == pytest.approx(100)

    def test_calories_away_from_target_score_lower(self):
        good = self.manager.score_response(response_set("A", response_data(self.request)), self.request)
        poor = self.manager.score_response(
            response_set("B", response_data(self.request, calorie_factor=0.5)), self.request)
        assert poor.score < good.score
        assert poor.components["calories"] == pytest.approx(0.5)

    def test_an_over_budget_day_lowers_the_budget_component(self):
        request = make_request(make_profile(), daily_budget=3)
        result = self.manager.score_response(response_set("A", response_data(request, cost=3.0)), request)
        assert result.components["budget"] < 1.0

    def test_unknown_costs_are_never_treated_as_free(self):
        data = response_data(self.request)
        data["recommendations"][0]["estimated_cost"] = None
        assert self.manager.score_response(response_set("A", data), self.request).components["budget"] == 0.0

    def test_duplicate_dishes_lower_the_variety_component(self):
        data = response_data(self.request)
        for item in data["recommendations"]:
            item["meal_name"] = "Same Dish"
        assert self.manager.score_response(response_set("A", data), self.request).components["variety"] == pytest.approx(0.1)

    def test_missing_recipe_steps_lower_the_practicality_component(self):
        data = response_data(self.request)
        for item in data["recommendations"][:5]:
            item["recipe_steps"] = []
        assert self.manager.score_response(response_set("A", data), self.request).components["practicality"] == pytest.approx(0.5)

    # -- hard constraints: each one makes the whole response ineligible ---------------------
    @pytest.mark.parametrize("count", [9, 11])
    def test_a_response_with_the_wrong_number_of_meals_is_rejected(self, count):
        data = response_data(self.request)
        data["recommendations"] = (data["recommendations"] * 2)[:count]
        assert self.manager.score_response(response_set("A", data), self.request) is None

    def test_one_unsafe_dish_rejects_the_entire_response(self):
        request = make_request(make_profile(dietary_restrictions=["peanut"]))
        data = response_data(request)
        data["recommendations"][3]["ingredients"] = ["peanut butter"]
        assert self.manager.score_response(response_set("A", data), request) is None

    def test_a_dish_with_no_ingredients_is_rejected_because_safety_is_unknown(self):
        data = response_data(self.request)
        data["recommendations"][0]["ingredients"] = []
        assert self.manager.score_response(response_set("A", data), self.request) is None

    def test_eat_out_dishes_are_rejected_when_the_user_wants_to_eat_in(self):
        request = make_request(make_profile(), meal_source="in")
        assert self.manager.score_response(response_set("A", sample_response()), request) is None

    def test_a_response_that_cannot_fill_every_slot_is_rejected(self):
        data = response_data(self.request)
        for item in data["recommendations"]:
            if item["meal_type"] == "dinner":
                item["meal_type"] = "lunch"
        assert self.manager.score_response(response_set("A", data), self.request) is None


class TestSelectingAiResponses:
    manager = LogicManager()
    request = make_request(make_profile())

    def good(self, provider="Gemini"):
        return response_set(provider, response_data(self.request))

    def poor(self, provider="Claude"):
        return response_set(provider, response_data(self.request, calorie_factor=0.5))

    def test_the_better_suited_response_wins(self):
        assert self.manager.select_response_sets([self.poor("Claude"), self.good("Gemini")], self.request) \
            .response.provider == "Gemini"

    def test_the_order_responses_arrive_in_does_not_matter(self):
        first = self.manager.select_response_sets([self.good("Gemini"), self.poor("Claude")], self.request)
        second = self.manager.select_response_sets([self.poor("Claude"), self.good("Gemini")], self.request)
        assert first.response.provider == second.response.provider == "Gemini"

    def test_an_ineligible_response_is_skipped_in_favour_of_a_valid_one(self):
        invalid = response_set("Claude", {"recommendations": response_data(self.request)["recommendations"][:5]})
        assert self.manager.select_response_sets([invalid, self.poor("Gemini")], self.request) \
            .response.provider == "Gemini"

    def test_when_all_are_ineligible_no_suitable_meals_is_raised(self):
        invalid = response_set("Claude", {"recommendations": response_data(self.request)["recommendations"][:5]})
        with pytest.raises(NoSuitableMealsError):
            self.manager.select_response_sets([invalid], self.request)

    def test_no_responses_at_all_raises_no_suitable_meals(self):
        with pytest.raises(NoSuitableMealsError):
            self.manager.select_response_sets([], self.request)

    def test_equal_scores_are_settled_by_the_cheaper_day(self):
        request = make_request(make_profile(), daily_budget=None)
        dear = response_set("Claude", response_data(request, cost=4.0))
        cheap = response_set("Gemini", response_data(request, cost=2.0))
        assert self.manager.select_response_sets([dear, cheap], request).response.provider == "Gemini"

    def test_identical_responses_fall_back_to_the_provider_name_so_the_result_is_stable(self):
        a, b = self.good("Gemini"), self.good("Claude")
        assert self.manager.select_response_sets([a, b], self.request).response.provider == "Claude"
        assert self.manager.select_response_sets([b, a], self.request).response.provider == "Claude"

    def test_the_winning_plan_contains_only_the_winning_providers_dishes(self):
        winner = self.manager.select_response_sets([self.poor("Claude"), self.good("Gemini")], self.request)
        winning_names = {c.meal_name for c in winner.response.candidates}
        assert {m.name for m in winner.plan.meals.values()} <= winning_names


# =============================================================================================
# Hand-off from the AI Manager: raw AI text -> validated candidates -> plan
# =============================================================================================
class TestAiTextToPlan:
    handler = AIHandlerManager()
    manager = LogicManager()

    def test_hardcoded_ai_json_becomes_a_complete_plan(self):
        text = json.dumps(sample_response())
        candidates = self.handler.parse_response(text)
        plan = self.manager.build_meal_plan(candidates, make_request(make_profile()))
        assert len(candidates) == config.RECOMMENDATION_COUNT
        assert plan.is_complete and set(plan.to_dict()["meals"]) == set(MealPlan.SLOTS)

    def test_dietary_safety_applies_to_parsed_ai_output(self):
        data = sample_response()
        for item in data["recommendations"]:
            item["ingredients"] = ["peanut"]
        candidates = self.handler.parse_response(json.dumps(data))
        request = make_request(make_profile(dietary_restrictions=["peanut"]))
        assert self.manager.build_meal_plan(candidates, request) is None

    @pytest.mark.parametrize("text", ["", "not json at all", "[]", '{"recommendations": []}', "null"])
    def test_malformed_ai_output_is_rejected_before_it_reaches_the_logic(self, text):
        with pytest.raises(AIResponseError):
            self.handler.parse_response(text)

    def _break(self, change):
        data = sample_response()
        change(data)
        return json.dumps(data)

    @pytest.mark.parametrize("change", [
        lambda d: d["recommendations"].pop(),                                        # 9 instead of 10
        lambda d: d["recommendations"][0].update(calories=-5),                       # negative number
        lambda d: d["recommendations"][0].update(calories="400"),                    # number as text
        lambda d: d["recommendations"][0].update(meal_source_type="delivery"),       # unknown source
        lambda d: d["recommendations"][0].update(meal_type="brunch"),                # unknown slot
        lambda d: d["recommendations"][0].pop("ingredients"),                        # missing field
        lambda d: d.update(extra="field"),                                           # unexpected top-level field
        lambda d: d["estimated_nutrition_targets"].pop("calorie_target"),            # incomplete targets
    ])
    def test_schema_violations_are_rejected(self, change):
        with pytest.raises(AIResponseError):
            self.handler.parse_response(self._break(change))
