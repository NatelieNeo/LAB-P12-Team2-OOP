from typing import Any, Callable, Dict, List, Mapping, Optional
from wsgiref.validate import validator

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

    def validate_number(self, raw: Any, label: str, low: float, high: float) -> float:
        try:
            value = float(str(raw).strip())
        except ValueError:
            raise InvalidInputError(f"Please enter your {label} as a number.") from None
        if not low < value < high:
            raise InvalidInputError(f"Please enter a realistic {label} between {low} and {high}.")
        return value

    def validate_weight(self, raw: Any) -> float:
        return self.validate_number(raw, "weight", 20, 300)

    def validate_height(self, raw: Any) -> float:
        return self.validate_number(raw, "height", 50, 250)

    def validate_activity_level(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value not in self.ACTIVITY_LEVELS:
            raise InvalidInputError(
                f"Please enter one of the following activity levels: {', '.join(self.ACTIVITY_LEVELS)}."
            )
        return value

    def validate_goal(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value not in self.GOALS:
            raise InvalidInputError(
                f"Please enter one of the following goals: {', '.join(self.GOALS)}."
            )
        return value

    def validate_restrictions(self, raw: Any) -> List[str]:
        text = str(raw or "").strip().upper()
        if text.isdigit():
            raise InvalidInputError("Please enter dietary restrictions as text, not numbers.")
        return UserProfile.normalize_restrictions(text)

    def validate_meal_preference(self, raw: Any) -> str:
        text = str(raw or "").strip().upper()
        if text.isdigit():
            raise InvalidInputError("Please enter meal preference as text, not numbers.")   
        return text

    def validate_meal_source(self, raw: Any) -> str:
        value = str(raw or "").strip().upper()
        if value not in self.MEAL_SOURCES:
            raise InvalidInputError(
                f"Please enter one of the following meal sources: {', '.join(self.MEAL_SOURCES)}."
            )
        return value

    def validate_budget(self, raw: Any) -> Optional[float]:
        #blank/0 means no budget
        text = str(raw if raw is not None else "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            raise InvalidInputError("Please enter your budget as a number.") from None
        if value == 0:
            return None
        if value < self.MIN_BUDGET:
            raise InvalidInputError(f"Please enter a budget of at least ${self.MIN_BUDGET}.")   
        return value

    def parse_list(self, raw: Any) -> List[str]:
        return [item.strip() for item in str(raw or "").split(",") if item.strip()]


class IOManager:
    #reads inputs and shows output for the terminal, turns web form data into model objects


    ALLERGEN_NOTICE = ("Always check food labels and ask the stall about allergens. "
                        "Calora cannot guarantee the absence of allergens.")
    MENU = ("1. Create a meal plan\n"
            "2. View your previous meal plans\n"
            "3. View your profile\n"
            "4. Update your profile\n"
            "5. Exit\n")
    SLOT_ORDER = MealPlan.SLOTS

    def __init__(self, validator: Optional[UserInputValidator] = None,
        input_fn: Callable[[str], str] = input, output_fn: Callable[..., None] = print):
        self._validator = validator or UserInputValidator()
        self._input = input_fn
        self._output = output_fn


    #terminal input

    def ask_user(self, prompt: str, validate: Callable[[str], Any], allow_blank: bool = False) -> Any:
        while True:
            raw = self._input(prompt)
            if allow_blank and raw.strip() == "":
                return None
            try:
                return validate(raw)
            except InvalidInputError as error:
                self._output(f"Error: {error}") 

    def welcome(self) -> None:
        self._output("Welcome to Calora! Your personal meal planning assistant.\n")
        self._output("I can help you track your health and fitness goals.\n")
        self._output(self.ALLERGEN_NOTICE)
                      

        