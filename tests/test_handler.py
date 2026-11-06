"""AIHandlerManager: parsing and schema validation. No network involved."""
import json

import pytest

from manager.ai_handler_manager import AIHandlerManager
from manager.exceptions import AIResponseError
from manager.models import MealCandidate
from tests.conftest import meal


@pytest.fixture
def handler():
    return AIHandlerManager()


def test_valid_response_becomes_candidates(handler, response):
    candidates = handler.parse_response(" \n" + json.dumps(response) + "\t ")
    assert len(candidates) == 10
    assert all(isinstance(c, MealCandidate) for c in candidates)
    assert candidates[0].meal_name == "Eggs on Toast"
    assert handler.validate_response(response)


@pytest.mark.parametrize("text", [None, "", " \n\t", "{", '{"recommendations":}', "null", "[]", "{}", "not JSON"])
def test_invalid_text_raises_ai_response_error(handler, text):
    with pytest.raises(AIResponseError):
        handler.parse_response(text)


@pytest.mark.parametrize("wrapper", ["```json\n{}\n```", "```\n{}\n```", "Here is your plan: {}", "{} trailing text"])
def test_markdown_fences_and_prose_are_rejected(handler, response, wrapper):
    with pytest.raises(AIResponseError):
        handler.parse_response(wrapper.format(json.dumps(response)))


@pytest.mark.parametrize("value", [123, {"recommendations": []}, [1]])
def test_non_text_input_raises_type_error(handler, value):
    with pytest.raises(TypeError):
        handler.parse_response(value)


@pytest.mark.parametrize("section", ["top", "targets", "meal"])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_schema_requires_exact_fields(handler, response, section, operation):
    target = {"top": response, "targets": response["estimated_nutrition_targets"],
              "meal": response["recommendations"][0]}[section]
    if operation == "missing":
        for field in list(target):
            value = target.pop(field)
            with pytest.raises(AIResponseError):
                handler.parse_response(json.dumps(response))
            target[field] = value
    else:
        target["unexpected"] = "extra"
        with pytest.raises(AIResponseError):
            handler.parse_response(json.dumps(response))


@pytest.mark.parametrize("field", ["meal_type", "meal_name", "portion_size", "meal_source_type", "calories",
                                   "protein_g", "carbs_g", "fat_g", "estimated_cost",
                                   "pantry_ingredients_used", "ingredients", "recipe_steps"])
def test_wrong_field_types_are_rejected(handler, response, field):
    response["recommendations"][0][field] = None
    assert not handler._validate_recommendation(response["recommendations"][0])
    with pytest.raises(AIResponseError):
        handler.parse_response(json.dumps(response))


@pytest.mark.parametrize("field", ["calorie_target", "protein_target_g", "carb_target_g", "fat_target_g"])
def test_wrong_target_types_are_rejected(handler, response, field):
    response["estimated_nutrition_targets"][field] = "100"
    assert not handler._validate_targets(response["estimated_nutrition_targets"])
    with pytest.raises(AIResponseError):
        handler.parse_response(json.dumps(response))


@pytest.mark.parametrize("count", [0, 1, 9, 10, 11])
def test_exactly_ten_recommendations(handler, count):
    assert handler._validate_recommendations([meal() for _ in range(count)]) is (count == 10)


def test_expected_count_is_configurable(response):
    assert AIHandlerManager(expected_count=3)._validate_recommendations([meal()] * 3)


@pytest.mark.parametrize("value", [None, "", 3, [], {}])
def test_container_shapes_are_rejected(handler, value):
    assert not handler._validate_targets(value)
    assert not handler._validate_recommendation(value)
    assert not handler._validate_recommendations(value)
    assert not handler.validate_response(value)


def test_known_validator_limitations(handler, response):
    """Documented behaviour: values are type-checked, not range-checked."""
    item = response["recommendations"][0]
    item.update(calories=True, protein_g=-1, ingredients=[123], meal_type="Snack")
    assert handler._validate_recommendation(item)
    item["meal_source_type"] = "in"
    assert not handler._validate_recommendation(item)


def test_invalid_response_is_logged(handler, caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(AIResponseError):
            handler.parse_response("{")
    assert "invalid JSON" in caplog.text
