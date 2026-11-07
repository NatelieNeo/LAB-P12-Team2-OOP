from typing import Any, Callable, Dict, List, Mapping, Optional

from manager.exceptions import InvalidInputError
from manager.models import MealPlan, MealPlanRequest, ShoppingList, UserProfile


class UserInputValidator:
    """Checks and cleans raw text. Each method returns a clean value or raises
    InvalidInputError. The CLI and the web form share these rules."""

    ACTIVITY_LEVELS = ("sedentary", "moderately active", "very active")
    GOALS = ("lose", "maintain", "gain")
    MEAL_SOURCES = ("in", "out", "both")
    MIN_BUDGET = 6

    def validate_username(self, raw: Any) -> str:
        value = str(raw or "").strip()
        if not value:
            raise InvalidInputError("Please enter a valid username.")
        return value

    def validate_name(self, raw: Any) -> str:
        value = str(raw or "").strip()
        if not value:
            raise InvalidInputError("Please enter a valid name.")
        return value

    def validate_age(self, raw: Any) -> int:
        try:
            age = int(str(raw or "").strip())
        except ValueError:
            raise InvalidInputError("Please enter a whole number.")
        if not 12 < age < 100:
            raise InvalidInputError("Please enter a valid age between 13 and 99.")
        return age

    def validate_gender(self, raw: Any) -> str:
        value = str(raw or "").strip().upper()
        if value not in ("M", "F"):
            raise InvalidInputError("Please enter M or F for gender.")
        return "male" if value == "M" else "female"

    
