
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