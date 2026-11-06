"""NEW FEATURE: shopping list for a plan's home-cooked meals."""
from manager.logic_manager import LogicManager, ShoppingListBuilder
from tests.conftest import make_profile, make_request


def test_builds_list_of_ingredients_not_in_pantry(candidates, request_both):
    manager = LogicManager()
    plan = manager.build_meal_plan(candidates, request_both)
    shopping = manager.build_shopping_list(plan, request_both.pantry_ingredients)
    assert shopping.items == ["2 slices wholemeal bread"]
    assert shopping.covered_by_pantry == ["2 eggs", "150g chicken breast", "1 cup cooked rice"]


def test_empty_pantry_means_everything_is_bought(candidates, request_both):
    manager = LogicManager()
    plan = manager.build_meal_plan(candidates, request_both)
    shopping = manager.build_shopping_list(plan, [])
    assert shopping.items == ["2 eggs", "2 slices wholemeal bread", "150g chicken breast", "1 cup cooked rice"]
    assert shopping.covered_by_pantry == []


def test_pantry_match_ignores_case_and_matches_substrings(candidates, request_both):
    plan = LogicManager().build_meal_plan(candidates, request_both)
    shopping = ShoppingListBuilder().build(plan, ["EGGS", " Chicken ", "BREAD", "rice"])
    assert shopping.is_empty


def test_eat_out_meals_need_no_shopping(candidates):
    request = make_request(make_profile(), meal_source="out", pantry_ingredients=[])
    plan = LogicManager().build_meal_plan(candidates, request)
    assert all(m.source_type == "eat_out" for m in plan.meals.values())
    assert ShoppingListBuilder().build(plan, []).is_empty


def test_duplicate_ingredients_are_listed_once(candidates, request_both):
    request = make_request(make_profile(), meal_source="in", pantry_ingredients=[])
    plan = LogicManager().build_meal_plan(candidates, request)
    items = ShoppingListBuilder().build(plan, []).items
    assert len(items) == len({i.lower() for i in items})


def test_object_style_ingredients_show_quantity_and_unit(request_both):
    from manager.models import MealCandidate
    from tests.conftest import meal
    item = MealCandidate.from_dict(meal(ingredients=[{"item": "rice", "quantity": 100, "unit": "g"}]))
    request = make_request(make_profile(), pantry_ingredients=[])
    plan = LogicManager().build_meal_plan([item], request)
    assert ShoppingListBuilder().build(plan, []).items == ["100 g rice"]
