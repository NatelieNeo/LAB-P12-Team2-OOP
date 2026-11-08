from typing import Any, Callable, Dict, List, Mapping, Optional
from wsgiref.validate import validator

from altair import value

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

    def read_username(self, raw: Any) -> str: #validate username from web form
        return self._validator.validate_username(raw)    

    def prompt_username(self) -> str: #validate username from terminal
        return self.ask_user("Enter your username: ", self._validator.validate_username)

    def prompt_profile(self, username: str) -> UserProfile:
        v = self._validator
        name = self._ask("Please enter your name: ", v.validate_name)
        self._output(f"Hello, {name}! Nice to meet you.\n")
        age = self._ask("Please enter your age: ", v.validate_age)
        gender = self._ask("Please enter your gender (M/F): ", v.validate_gender)
        weight = self._ask("Please enter your weight (kg): ", v.validate_weight)
        height = self._ask("Please enter your height (cm): ", v.validate_height)
        activity_level = self._ask("Please enter your activity level (sedentary, moderately active, very active): ", v.validate_activity_level)
        goal = self._ask("Please enter your goal (lose, maintain, gain): ", v.validate_goal)
        restrictions = self._ask("Please enter any dietary restrictions (comma-separated, or leave blank): ", v.validate_restrictions, allow_blank=True)
        return UserProfile(
            username=username,
            name=name,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            goal=goal,
            restrictions=restrictions,
            
        )

    def prompt_profile_changes(self, current: UserProfile) -> Dict[str,Any]:
    #ask for each field agn, nil means no change
        v = self._validator
        self._output("\nUpdate your profile. Leave blank to keep current value.\n")    
        fields = (
            ("name", "Please enter your name: ", v.validate_name),
            ("age", "Please enter your age: ", v.validate_age),
            ("gender", "Please enter your gender (M/F): ", v.validate_gender),
            ("weight", "Please enter your weight (kg): ", v.validate_weight),
            ("height", "Please enter your height (cm): ", v.validate_height),
            ("activity_level", "Please enter your activity level (sedentary, moderately active, very active): ", v.validate_activity_level),
            ("goal", "Please enter your goal (lose, maintain, gain): ", v.validate_goal),
            ("dietary_restrictions", "Please enter any dietary restrictions (comma-separated, or leave blank): ", v.validate_restrictions)
        )
        changes = {}
        for field, prompt, validate in fields:
            answer = self._ask(prompt, validate, allow_blank=True)
            if answer is not None:
                changes[field] = answer
        return changes

    def prompt_request(self, profile: UserProfile) -> MealPlanRequest:
        v = self._validator
        self._output("\nLet's create a meal plan for you.\n")
        meal_preference = self._ask("Please enter your meal preference (e.g., vegetarian, low-carb, etc.): ", v.validate_meal_preference)
        self._output(f"Great! You prefer {meal_preference} meals.\n")
        meal_source = self._ask("Do you want to eat in, out, or both? Enter in/out/both ", v.validate_meal_source)
        self._output(f"Got it! You want to eat {meal_source}.\n")
        pantry: List[str] = []
        if meal_source != "out": #only if eating in or both, ask for pantry items
            pantry = v.parse_list(self._input("Please enter your pantry items (comma-separated, or leave blank): "))
            self._output(f"Thanks! You have {len(pantry)} items in your pantry.\n")
        budget = self._ask(f"Please enter your budget (minimum ${v.MIN_BUDGET}) (or leave blank for no budget): ",
        v.validate_budget, allow_blank=True)
        self._output(f"Got it! Your budget is {budget}.\n")
        return MealPlanRequest(
            profile=profile,
            meal_source=meal_source,
            budget=budget,
            meal_preference=meal_preference,
            pantry=pantry
        )
    
    def prompt_menu_choice(self) -> str:
        for line in self.MENU:
            self._output(line)
        return self._input("\nPlease enter your choice (1-5): ").strip()
        
    def pause(self) -> None:
        self._input("\nPress Enter to return to the main menu...")

    # terminal output
    def show_message(self, message: Any) -> None:
        self._output(message)

    def allergen_notice(self, plan: MealPlan) -> str: #disclaimer for eating out, if any meal is from outside
        if any(meal.source_type == "eat_out" for meal in plan.meals.values()):

            return self.ALLERGEN_NOTICE
        return ""

    def format_plan(self, plan: MealPlan) -> str:
        data = plan.to_dict()
        lines = ["=== Your Meal Plan ===", f"Why this plan: {data['explanation']} "]

        for slot in self.SLOT_ORDER:
            meal = data["meals"].get(slot)
            if meal is None:
                continue
            lines.extend([
                 "",
                    f"{slot.capitalize()}: {meal.name} ({meal.source_type})",
                    f"Calories: {meal['calorie_range']} {meal['calorie_tag']} for your target",                       
                    f"Macros: Protein {meal['protein_tag']} / Carbs {meal['carbs_tag']} / Fat {meal['fat_tag']}" 
                            
            ])

            if meal["pantry_ingredients_used"]:
                lines.append("Ingredients: " + ", ".join(plan.meals[slot].ingredients_labels))
            lines.append(f"Cost: {meal['cost_tier']}")
        if data["missing_slots"]:
            lines.extend(["","Missing meals: " + ", ".join(data["missing_slots"])])
        notice = self.allergen_notice(plan)
        if notice:
            lines.extend(["", notice])
        return "\n".join(lines)

    
                    