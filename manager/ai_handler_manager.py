
import json
import logging
from typing import Any, List

from config import RECOMMENDATION_COUNT
from manager.exceptions import AIResponseError
from manager.models import MealCandidate

logger = logging.getLogger(__name__)

class AIHandlerManager:
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
        return isinstance(value, (int, float))

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
        return (
            all(isinstance(recommendation[f], str) for f in self.STRING_FIELDS)
            and all(self._is_number(recommendation[f]) for f in self.NUMERIC_FIELDS)
            and all(isinstance(recommendation[f], list) for f in self.LIST_FIELDS)
        )