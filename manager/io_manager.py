"""IO Manager: everything the user types or reads (terminal and web form)."""
from typing import Any, Callable, Dict, List, Mapping, Optional

from manager.exceptions import InvalidInputError
from manager.models import MealPlan, MealPlanRequest, Recipe, ShoppingList, UserProfile


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
            raise InvalidInputError("Please enter your age as a whole number.") from None
        if not 12 < age < 100:
            raise InvalidInputError("Please enter a valid age between 13 and 99.")
        return age

    def validate_gender(self, raw: Any) -> str:
        value = str(raw or "").strip().upper()
        if value not in ("M", "F"):
            raise InvalidInputError("Please enter M or F for gender.")
        return "male" if value == "M" else "female"

    def _validate_number(self, raw: Any, label: str, low: float, high: float) -> float:
        try:
            value = float(str(raw).strip())
        except ValueError:
            raise InvalidInputError(f"Please enter your {label} as a number.") from None
        if not low < value < high:
            raise InvalidInputError(f"Please enter a realistic {label} (between {low:g} and {high:g}).")
        return value

    def validate_weight(self, raw: Any) -> float:
        return self._validate_number(raw, "weight in kg", 20, 300)

    def validate_height(self, raw: Any) -> float:
        return self._validate_number(raw, "height in cm", 50, 250)

    def validate_activity_level(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value not in self.ACTIVITY_LEVELS:
            raise InvalidInputError("Please choose: " + ", ".join(self.ACTIVITY_LEVELS) + ".")
        return value

    def validate_goal(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value not in self.GOALS:
            raise InvalidInputError("Please choose: " + ", ".join(self.GOALS) + ".")
        return value

    def validate_restrictions(self, raw: Any) -> List[str]:
        text = str(raw or "").strip()
        if text.isdigit():
            raise InvalidInputError("You entered a number. Please describe your restrictions in words.")
        return UserProfile.normalize_restrictions(text)

    def validate_meal_preference(self, raw: Any) -> str:
        text = str(raw or "").strip()
        if text.isdigit():
            raise InvalidInputError("You entered a number. Please describe the food you want in words.")
        return text

    def validate_meal_source(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value not in self.MEAL_SOURCES:
            raise InvalidInputError("Please enter in, out or both.")
        return value

    def validate_budget(self, raw: Any) -> Optional[float]:
        """Blank or 0 means no budget (None)."""
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
            raise InvalidInputError(f"The minimum daily budget is ${self.MIN_BUDGET} (enter 0 for none).")
        return value

    def parse_list(self, raw: Any) -> List[str]:
        return [item.strip() for item in str(raw or "").split(",") if item.strip()]


class IOManager:
    """Reads input and shows output for the terminal, and turns web form data
    into model objects. The input and output functions are injected so the
    class can be tested without a keyboard."""

    ALLERGEN_NOTICE = ("Always check food labels and ask the stall about allergens. "
                       "Calora cannot guarantee a meal is allergen-free.")
    MENU = ("1. Create a meal plan", "2. View previous meal plans", "3. View your profile",
            "4. Update your profile", "5. Exit the program",
            "R. View recipes from saved plans")
    SLOT_ORDER = MealPlan.SLOTS

    def __init__(self, validator: Optional[UserInputValidator] = None,
                 input_fn: Callable[[str], str] = input, output_fn: Callable[..., None] = print):
        self._validator = validator or UserInputValidator()
        self._input = input_fn
        self._output = output_fn

    # ---- terminal input ----------------------------------------------------
    def _ask(self, prompt: str, validate: Callable[[str], Any], allow_blank: bool = False) -> Any:
        """Keep asking until the answer is valid. A blank answer returns None
        when allow_blank is set (used when updating a profile)."""
        while True:
            raw = self._input(prompt)
            if allow_blank and raw.strip() == "":
                return None
            try:
                return validate(raw)
            except InvalidInputError as error:
                self._output(error)

    def welcome(self) -> None:
        self._output("Hi I am your personal health assistant, Calora!")
        self._output("I can help you track your health and fitness goals.\n")

    def read_username(self, raw: Any) -> str:
        """Validate a username typed into the web sign-in form."""
        return self._validator.validate_username(raw)

    def prompt_username(self) -> str:
        return self._ask("Please enter your username: ", self._validator.validate_username)

    def prompt_profile(self, username: str) -> UserProfile:
        v = self._validator
        name = self._ask("Please enter your name: ", v.validate_name)
        self._output(f"Hi {name}, it's nice to meet you!")
        age = self._ask("Please enter your age: ", v.validate_age)
        gender = self._ask("Please enter your gender (M/F): ", v.validate_gender)
        weight = self._ask("Please enter your weight in kg (up to 1 decimal place): ", v.validate_weight)
        height = self._ask("Please enter your height in cm (up to 1 decimal place): ", v.validate_height)
        activity = self._ask(
            "Please enter your activity level (sedentary /workout 1-2 times a week/, "
            "moderately active /workout 3-4 times a week/, very active /5-6 times a week/): ",
            v.validate_activity_level)
        goal = self._ask("Please enter your desired goal (lose/maintain/gain): ", v.validate_goal)
        restrictions = self._ask(
            "Please indicate your dietary restrictions (If none please enter 'nil'): ", v.validate_restrictions)
        return UserProfile(username, name, age, gender, weight, height, activity, goal, restrictions)

    def prompt_profile_changes(self, current: UserProfile) -> Dict[str, Any]:
        """Ask for each field again; a blank answer keeps the current value."""
        v = self._validator
        self._output("\nUpdate your profile information. Leave blank to keep current value.")
        fields = (
            ("name", "Please enter your name: ", v.validate_name),
            ("age", "Please enter your age: ", v.validate_age),
            ("gender", "Please enter your gender (M/F): ", v.validate_gender),
            ("weight_kg", "Please enter your weight in kg: ", v.validate_weight),
            ("height_cm", "Please enter your height in cm: ", v.validate_height),
            ("activity_level", "Please enter your activity level: ", v.validate_activity_level),
            ("goal", "Please enter your desired goal (lose/maintain/gain): ", v.validate_goal),
            ("dietary_restrictions", "Please indicate your dietary restrictions: ", v.validate_restrictions),
        )
        changes = {}
        for field, prompt, validate in fields:
            answer = self._ask(prompt, validate, allow_blank=True)
            if answer is not None:
                changes[field] = answer
        return changes

    def prompt_request(self, profile: UserProfile) -> MealPlanRequest:
        v = self._validator
        self._output(f"\nOkay {profile.name}, let's get started with today's meal plan!")
        preference = self._ask("What type of food do you want to eat today?: ", v.validate_meal_preference)
        self._output(f"Yum! {preference} sounds good.")
        source = self._ask("Are you planning to eat out or in today? enter in/out/both: ", v.validate_meal_source)
        pantry: List[str] = []
        if source != "out":  # no pantry needed when only eating out
            pantry = v.parse_list(self._input("What ingredients do you have?: "))
        budget = self._ask(
            f"What's your total budget for the day? (minimum ${v.MIN_BUDGET}) (enter '0' for none): ",
            v.validate_budget)
        return MealPlanRequest(profile, preference, source, pantry, budget)

    def prompt_menu_choice(self) -> str:
        for line in self.MENU:
            self._output(line)
        return self._input("\nPlease select an option from the list below (1-5 or R): ").strip().upper()

    def pause(self) -> None:
        self._input("\nPress Enter to return to the main menu...")

    # ---- terminal output -----------------------------------------------------
    def show_message(self, message: Any) -> None:
        self._output(message)

    def allergen_notice(self, plan: MealPlan) -> str:
        """The disclaimer, only when the plan has an eat-out meal (a home-cooked
        meal has no stall to ask)."""
        if any(meal.source_type == "eat_out" for meal in plan.meals.values()):
            return self.ALLERGEN_NOTICE
        return ""

    def format_plan(self, plan: MealPlan) -> str:
        data = plan.to_dict()
        lines = ["=== Your Meal Plan ===", f"Why this plan: {data['explanation']}"]
        for slot in self.SLOT_ORDER:
            meal = data["meals"].get(slot)
            if meal is None:
                continue
            lines.extend([
                "",
                f"{slot.capitalize()}: {meal['name']} ({meal['source_type']})",
                f"Calories: {meal['calorie_range']} — {meal['calorie_tag']} for your target",
                f"Macros: Protein {meal['protein_tag']} / Carbs {meal['carbs_tag']} / Fat {meal['fat_tag']}",
            ])
            if meal["pantry_ingredients_used"]:
                lines.append("Ingredients: " + ", ".join(plan.meals[slot].ingredient_labels))
            lines.append(f"Cost: {meal['cost_tier']}")
        if data["missing_slots"]:
            lines.extend(["", "Missing meals: " + ", ".join(data["missing_slots"])])
        notice = self.allergen_notice(plan)
        if notice:
            lines.extend(["", notice])
        return "\n".join(lines)

    def format_profile(self, profile: UserProfile) -> str:
        data = profile.to_dict(include_preferences=False)
        data["dietary_restrictions"] = ", ".join(data["dietary_restrictions"]) or "none"
        return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in data.items())

    def show_plan(self, plan: MealPlan) -> None:
        self._output(self.format_plan(plan))

    def show_history(self, plans: Mapping[str, MealPlan]) -> None:
        for entry_id, plan in plans.items():
            self._output(f"\nMeal Plan ID: {entry_id}\n{self.format_plan(plan)}")

    def show_profile(self, profile: UserProfile) -> None:
        self._output(f"\nHere is your profile information:\n{self.format_profile(profile)}")

    def format_recipes(self, recipes: List[Recipe]) -> str:
        """Step-by-step instructions for the home-cooked meals; empty text if there are none."""
        if not recipes:
            return ""
        blocks = [recipe.to_text() for recipe in recipes]
        return "=== Step-by-step recipes ===\n\n" + "\n\n".join(blocks)

    def show_recipes(self, recipes: List[Recipe]) -> None:
        text = self.format_recipes(recipes)
        if text:
            self._output("\n" + text)

    def show_shopping_list(self, shopping_list: ShoppingList) -> None:
        self._output("\n" + shopping_list.to_text())

    # ---- web forms -------------------------------------------------------------
    def read_profile_form(self, form: Mapping[str, str], username: Optional[str] = None) -> UserProfile:
        """Validate the profile part of a web form. Raises InvalidInputError."""
        v = self._validator
        return UserProfile(
            username=v.validate_username(username if username is not None else form.get("username")),
            name=v.validate_name(form.get("name")),
            age=v.validate_age(form.get("age")),
            gender=v.validate_gender(form.get("gender")),
            weight_kg=v.validate_weight(form.get("weight_kg")),
            height_cm=v.validate_height(form.get("height_cm")),
            activity_level=v.validate_activity_level(form.get("activity_level")),
            goal=v.validate_goal(form.get("goal")),
            dietary_restrictions=v.validate_restrictions(form.get("dietary_restrictions")),
        )

    def read_request_form(self, form: Mapping[str, str], profile: UserProfile) -> MealPlanRequest:
        """Validate the preferences part of a web form. Raises InvalidInputError."""
        v = self._validator
        source = v.validate_meal_source(form.get("meal_source") or "both")
        pantry = [] if source == "out" else v.parse_list(form.get("ingredients"))
        return MealPlanRequest(
            profile,
            v.validate_meal_preference(form.get("meal_preference")),
            source, pantry, v.validate_budget(form.get("budget")),
        )
