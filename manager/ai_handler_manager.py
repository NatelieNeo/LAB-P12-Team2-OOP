
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==========================================
# Base Validator (Abstraction & Inheritance)
# ==========================================

class BaseValidator(ABC):

    @abstractmethod
    def validate(self, data):
        """Validate the given data."""
        pass

    def has_exact_fields(self, data, required_fields):
        """Check whether a dictionary has the exact required fields."""
        return (
            isinstance(data, dict)
            and set(data.keys()) == required_fields
        )

    def is_number(self, value):
        """Check whether a value is numeric, excluding booleans."""
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )


# ==========================================
# Nutrition Targets Validator
# ==========================================

class NutritionTargetsValidator(BaseValidator):

    REQUIRED_FIELDS = {
        "calorie_target",
        "protein_target_g",
        "carb_target_g",
        "fat_target_g"
    }

    def validate(self, targets):

        if not self.has_exact_fields(targets, self.REQUIRED_FIELDS):
            return False

        for value in targets.values():
            if not self.is_number(value):
                return False

        return True


# ==========================================
# Individual Recommendation Validator
# ==========================================

class RecommendationValidator(BaseValidator):

    REQUIRED_FIELDS = {
        "meal_type",
        "meal_name",
        "meal_source_type",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "estimated_cost",
        "portion_size",
        "pantry_ingredients_used",
        "ingredients",
        "recipe_steps"
    }

    STRING_FIELDS = {
        "meal_type",
        "meal_name",
        "portion_size"
    }

    NUMERIC_FIELDS = {
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "estimated_cost"
    }

    LIST_FIELDS = {
        "pantry_ingredients_used",
        "ingredients",
        "recipe_steps"
    }

    VALID_SOURCE_TYPES = ("eat_out", "home_cooked")

    def validate(self, recommendation):

        if not self.has_exact_fields(
            recommendation, self.REQUIRED_FIELDS
        ):
            return False

        # Validate string fields
        for field in self.STRING_FIELDS:
            if not isinstance(recommendation[field], str):
                return False

        # Validate meal source type
        if recommendation["meal_source_type"] not in self.VALID_SOURCE_TYPES:
            return False

        # Validate numeric fields
        for field in self.NUMERIC_FIELDS:
            if not self.is_number(recommendation[field]):
                return False

        # Validate list fields
        for field in self.LIST_FIELDS:
            if not isinstance(recommendation[field], list):
                return False

        return True


# ==========================================
# Recommendations List Validator
# ==========================================

class RecommendationsValidator(BaseValidator):

    REQUIRED_COUNT = 10

    def __init__(self):
        self.recommendation_validator = RecommendationValidator()

    def validate(self, recommendations):

        if not isinstance(recommendations, list):
            return False

        if len(recommendations) != self.REQUIRED_COUNT:
            return False

        for recommendation in recommendations:
            if not self.recommendation_validator.validate(recommendation):
                return False

        return True


# ==========================================
# Complete Response Validator
# ==========================================

class ResponseValidator(BaseValidator):

    REQUIRED_FIELDS = {
        "estimated_nutrition_targets",
        "recommendations"
    }

    def __init__(self):
        self.nutrition_validator = NutritionTargetsValidator()
        self.recommendations_validator = RecommendationsValidator()

    def validate(self, data):

        if not self.has_exact_fields(data, self.REQUIRED_FIELDS):
            return False

        if not self.nutrition_validator.validate(
            data["estimated_nutrition_targets"]
        ):
            return False

        if not self.recommendations_validator.validate(
            data["recommendations"]
        ):
            return False

        return True


# ==========================================
# Gemini Response Parser
# ==========================================

class GeminiResponseParser:

    def __init__(self):
        self.response_validator = ResponseValidator()

    def parse_response(self, response_text):
        """
        Convert Gemini's response text into JSON
        and validate its structure.

        Returns:
            dict if valid
            None if invalid
        """

        if not response_text:
            logger.warning("AI returned an empty response.")
            return None

        try:
            data = json.loads(response_text)

        except (json.JSONDecodeError, TypeError) as error:
            logger.warning(
                "AI response contains invalid JSON: %s",
                error
            )
            return None

        if not self.response_validator.validate(data):
            logger.warning(
                "AI response failed schema validation."
            )
            return None

        return data
