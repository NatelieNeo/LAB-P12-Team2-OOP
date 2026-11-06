"""Proof that the mandatory OOP concepts are real, not decorative."""
import inspect
from abc import ABC

import pytest

from manager.ai_manager import AIManager, AIProvider, GeminiProvider, StubAIProvider
from manager.data_manager import DataManager, JsonDataManager, SqliteDataManager
from manager.exceptions import (
    AIConnectionError, AIResponseError, CaloraError, InvalidInputError, NoSuitableMealsError, StorageError)
from manager.job_manager import ProgressSpinner
from manager.logic_manager import (
    CalorieFitCriterion, CostCriterion, DietarySafetyRule, LogicManager, MacroFitCriterion, MealRule,
    PreferenceCriterion, RankingCriterion, SourcePreferenceRule)
from manager.models import MealCandidate, MealPlan, PlannedMeal, UserProfile
from tests.conftest import make_profile, make_request, meal


# ---- abstract classes and interfaces cannot be instantiated -------------------------------
@pytest.mark.parametrize("abstract", [DataManager, AIProvider, MealRule, RankingCriterion])
def test_abstract_types_cannot_be_created(abstract):
    assert issubclass(abstract, ABC)
    with pytest.raises(TypeError):
        abstract()


def test_a_subclass_missing_an_abstract_method_cannot_be_created():
    class Incomplete(MealRule):
        pass
    with pytest.raises(TypeError):
        Incomplete()


def test_abstract_class_has_shared_behaviour_but_interface_has_none():
    assert not inspect.isabstract(DataManager.profile_exists)
    assert "profile_exists" in vars(DataManager)
    for interface in (AIProvider, MealRule, RankingCriterion):
        assert all(getattr(getattr(interface, name), "__isabstractmethod__", False)
                   for name in interface.__abstractmethods__)
        assert len(interface.__abstractmethods__) == 1


# ---- inheritance ---------------------------------------------------------------------------------
def test_inheritance_hierarchies():
    assert issubclass(JsonDataManager, DataManager) and issubclass(SqliteDataManager, DataManager)
    assert issubclass(ProgressSpinner, __import__("threading").Thread)
    for error in (InvalidInputError, AIConnectionError, AIResponseError, StorageError, NoSuitableMealsError):
        assert issubclass(error, CaloraError)


# ---- polymorphism: one client loop, many behaviours, no type checks -----------------------------
def test_logic_manager_treats_every_rule_alike(candidates):
    class OnlyLunch(MealRule):
        def apply(self, candidates, request):
            return [c for c in candidates if c.slot == "lunch"]

    request = make_request(make_profile())
    plan = LogicManager(rules=[DietarySafetyRule(), SourcePreferenceRule(), OnlyLunch()]).build_meal_plan(
        candidates, request)
    assert list(plan.meals) == ["lunch"] and plan.missing_slots == ["breakfast", "dinner"]


def test_new_ranking_criterion_needs_no_change_to_logic_manager(candidates):
    """Open/Closed: add behaviour by adding a class."""
    class HighestCalories(RankingCriterion):
        def score(self, candidate, targets, request):
            return -candidate.calories

    plan = LogicManager(criteria=[HighestCalories()]).build_meal_plan(candidates, make_request(make_profile()))
    assert plan.meals["lunch"].name == "Vegetable Rice Bowl"  # the 650 kcal option


def test_criteria_order_is_the_priority_order(candidates):
    request = make_request(make_profile())
    cost_first = LogicManager(criteria=[CostCriterion(), CalorieFitCriterion()])
    calorie_first = LogicManager(criteria=[CalorieFitCriterion(), CostCriterion()])
    assert cost_first.build_meal_plan(candidates, request).meals["lunch"].name == "Garlic Chicken Pasta"
    assert calorie_first.build_meal_plan(candidates, request).meals["lunch"].name == "Chicken Rice"


def test_every_criterion_has_the_same_signature(targets=None):
    signatures = {inspect.signature(c.score) for c in
                  (CalorieFitCriterion, MacroFitCriterion, PreferenceCriterion, CostCriterion)}
    assert len(signatures) == 1


def test_ai_manager_works_with_any_provider(response_text, request_both):
    from manager.logic_manager import NutritionCalculator
    targets = NutritionCalculator().targets(request_both.profile)

    class Recording(AIProvider):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return response_text

    custom = Recording()
    assert len(AIManager(custom).get_recommendations(request_both, targets)) == 10 and len(custom.prompts) == 1
    assert len(AIManager(StubAIProvider(response_text)).get_recommendations(request_both, targets)) == 10


def test_substituting_a_data_manager_does_not_change_results(tmp_path, candidates, request_both):
    plans = []
    for manager in (JsonDataManager(str(tmp_path / "u"), str(tmp_path / "p")), SqliteDataManager(str(tmp_path / "d.db"))):
        manager.save_plan(LogicManager().build_meal_plan(candidates, request_both), "u")
        plans.append(next(iter(manager.plans_for_user("u").values())).to_dict())
    assert plans[0] == plans[1]


# ---- encapsulation ----------------------------------------------------------------------------------
def test_private_state_is_not_reachable_by_its_plain_name():
    provider = GeminiProvider(api_key="k")
    assert not hasattr(provider, "__api_key") and not hasattr(provider, "api_key")
    assert hasattr(provider, "_GeminiProvider__api_key")


@pytest.mark.parametrize("cls,attribute", [(UserProfile, "age"), (MealCandidate, "calories"),
                                           (PlannedMeal, "name"), (MealPlan, "total_calories")])
def test_model_properties_have_no_setters(cls, attribute):
    assert isinstance(getattr(cls, attribute), property) and getattr(cls, attribute).fset is None


def test_models_hand_out_copies_not_internals():
    plan = MealPlan(1500)
    plan.meals["breakfast"] = "x"
    assert plan.meals == {}


# ---- composition vs aggregation --------------------------------------------------------------------
def test_composition_owner_creates_its_parts():
    manager = LogicManager()
    assert manager._calculator is not LogicManager()._calculator  # each manager builds its own
    assert type(manager._tagger).__name__ == "MealTagger"


def test_aggregation_parts_are_injected_and_outlive_the_whole():
    provider = StubAIProvider("x")
    ai = AIManager(provider)
    assert ai._provider is provider
    del ai
    assert provider.generate("hi") == "x"


# ---- association --------------------------------------------------------------------------------------
def test_request_refers_to_a_profile_without_owning_it():
    profile = make_profile()
    one, two = make_request(profile), make_request(profile, meal_source="out")
    assert one.profile is two.profile is profile


def test_plan_owns_planned_meals():
    plan = MealPlan(1500)
    planned = PlannedMeal("breakfast", "x", "home_cooked", "a", "b", "c", "d", "e", "f")
    plan.add_meal(planned)
    assert plan.meals["breakfast"] is planned
