"""AI Handler Manager: turns Gemini's raw text into validated MealCandidate objects."""
import json
import logging
import math
from typing import Any, List

from config import RECOMMENDATION_COUNT
from manager.exceptions import AIResponseError
from manager.models import MealCandidate

logger = logging.getLogger(__name__)


class AIHandlerManager:
    """Parses and validates the AI response. The schema is defined once, here,
    and every check below reads from it."""

    TARGET_FIELDS = {"calorie_target", "protein_target_g", "carb_target_g", "fat_target_g"}
    STRING_FIELDS = ["meal_type", "meal_name", "portion_size"]
    NUMERIC_FIELDS = ["calories", "protein_g", "carbs_g", "fat_g", "estimated_cost"]
    LIST_FIELDS = ["pantry_ingredients_used", "ingredients", "recipe_steps"]
    MEAL_SOURCE_TYPES = ("eat_out", "home_cooked")
    TOP_LEVEL_FIELDS = {"estimated_nutrition_targets", "recommendations"}
    RECOMMENDATION_FIELDS = set(STRING_FIELDS + NUMERIC_FIELDS + LIST_FIELDS + ["meal_source_type"])

    def __init__(self, expected_count: int = RECOMMENDATION_COUNT):
        self._expected_count = expected_count

    @staticmethod
    def _is_number(value: Any) -> bool:
        if type(value) not in (int, float) or value < 0:
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    def _validate_targets(self, targets: Any) -> bool:
        return (isinstance(targets, dict)
                and set(targets.keys()) == self.TARGET_FIELDS
                and all(self._is_number(value) for value in targets.values()))
    
    def _validate_recommendation(self, recommendation: Any) -> bool:
        if not isinstance(recommendation, dict):
            return False
        if set(recommendation.keys()) != self.RECOMMENDATION_FIELDS:
            return False
        if recommendation["meal_source_type"] not in self.MEAL_SOURCE_TYPES:
            return False
        if recommendation["meal_type"] not in ("breakfast", "lunch", "dinner"):
            return False
        if not isinstance(recommendation["meal_name"], str) or not recommendation["meal_name"].strip():
            return False
        return (
            all(isinstance(recommendation[f], str) for f in self.STRING_FIELDS)
            and all(self._is_number(recommendation[f]) for f in self.NUMERIC_FIELDS)
            and all(isinstance(recommendation[f], list) for f in self.LIST_FIELDS)
            and recommendation["calories"] > 0
            and all(isinstance(item, str) for item in recommendation["pantry_ingredients_used"])
            and all(isinstance(item, str) for item in recommendation["recipe_steps"])
            and all(self._valid_ingredient(item) for item in recommendation["ingredients"])
        )

    @staticmethod
    def _valid_ingredient(item: Any) -> bool:
        if isinstance(item, str):
            return bool(item.strip())
        return (isinstance(item, dict) and isinstance(item.get("item"), str)
                and bool(item["item"].strip()) and isinstance(item.get("unit", ""), str)
                and isinstance(item.get("quantity", ""), (int, float, str)))
    
    def _validate_recommendations(self, recommendations: Any) -> bool:
        return (isinstance(recommendations, list)
                and len(recommendations) == self._expected_count
                and all(self._validate_recommendation(r) for r in recommendations))

    def validate_response(self, data: Any) -> bool:
        """True only if the whole response matches the expected schema."""
        return (isinstance(data, dict)
                and set(data.keys()) == self.TOP_LEVEL_FIELDS
                and self._validate_targets(data["estimated_nutrition_targets"])
                and self._validate_recommendations(data["recommendations"]))
    
    def is_valid(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except AIResponseError:
            return False
        return True
    
    def parse_response(self, response_text: str) -> List[MealCandidate]:
        """Text -> validated MealCandidate objects.

        Raises AIResponseError for an empty, non-JSON or schema-invalid response."""
        if not isinstance(response_text, str) or not response_text:
            raise AIResponseError("The AI returned an empty response.")
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as error:
            logger.warning("AI response contains invalid JSON: %s", error)
            raise AIResponseError(f"The AI response was not valid JSON: {error}") from error
        if not self.validate_response(data):
            logger.warning("AI response failed schema validation.")
            raise AIResponseError("The AI response did not match the expected format.")
        return [MealCandidate.from_dict(item) for item in data["recommendations"]]
