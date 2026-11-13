"""Domain model classes for Calora.

Attributes are private or protected by naming convention (``__x`` / ``_x``)
and exposed through read-only properties, so other classes cannot put an
object into an inconsistent state by assigning to its fields directly.
"""
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AIResponseSet:
    """One validated provider response with monotonic worker timing."""

    provider: str
    candidates: Tuple["MealCandidate", ...]
    started_at: float
    completed_at: float


@dataclass(frozen=True)
class ResponseSelection:
    """LogicManager's winning complete set and its independently built plan."""

    response: AIResponseSet
    plan: "MealPlan"
    score: float
    components: Dict[str, float]


class IngredientParser:
    """Reads the AI's ingredient entries, each a plain string or an
    ``{"item": ..., "quantity": ..., "unit": ...}`` object."""

    MEASUREMENT_UNITS = {
        "g", "kg", "mg", "gram", "kilogram", "ml", "l", "litre", "liter",
        "tsp", "tbsp", "teaspoon", "tablespoon", "cup", "slice", "piece", "clove",
    }

    @classmethod
    def without_measurement(cls, label: str) -> str:
        """Format a shopping label as a name while leaving stored recipe labels intact."""
        text = str(label).strip()
        number = r"(?:(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?|[¼½¾⅓⅔⅛⅜⅝⅞])"
        amount = rf"{number}(?:\s*[-–]\s*{number}|\s*{number})?"
        units = "|".join(sorted(cls.MEASUREMENT_UNITS, key=len, reverse=True))
        measurement = rf"{amount}(?:\s*(?:{units})s?\b\.?)?"
        # AI strings and saved recipe labels can put quantities in brackets
        # before or after the name. Preserve non-measurement notes in brackets.
        name = re.sub(rf"(?:\(\s*{measurement}\s*\)|\[\s*{measurement}\s*\])", "", text,
                      flags=re.IGNORECASE)
        name = re.sub(rf"^\s*{amount}\s*(?:(?:{units})s?\b\.?\s*)?(?:of\s+)?", "", name,
                      flags=re.IGNORECASE)
        name = re.sub(rf"(?:\s*[-,:–]\s*|\s+){amount}\s*(?:{units})s?\b\.?\s*$", "", name,
                      flags=re.IGNORECASE)
        return name.strip() or text

    @staticmethod
    def names(ingredients: Optional[List[Any]]) -> List[str]:
        names = []
        for ingredient in ingredients or []:
            if isinstance(ingredient, dict):
                item = ingredient.get("item")
                if item:
                    names.append(str(item))
            elif isinstance(ingredient, str) and ingredient:
                names.append(ingredient)
        return names

    @staticmethod
    def labels(ingredients: Optional[List[Any]]) -> List[str]:
        """Human-readable text, including quantity and unit when given."""
        labels = []
        for ingredient in ingredients or []:
            if isinstance(ingredient, dict):
                parts = [ingredient.get("quantity"), ingredient.get("unit"), ingredient.get("item")]
                text = " ".join(str(part) for part in parts if part not in (None, ""))
                if text:
                    labels.append(text)
            elif isinstance(ingredient, str) and ingredient:
                labels.append(ingredient)
        return labels


class UserProfile:
    """A registered user: body data, goal and dietary restrictions."""

    _ACTIVITY_ALIASES = {"moderately_active": "moderate"}
    _NO_RESTRICTION_WORDS = {"", "nil", "none", "no", "no restrictions", "n/a", "na"}
    _RESTRICTION_PREFIXES = (
        "no ", "without ", "avoid ",
        "i am allergic to ", "i'm allergic to ", "allergic to ", "allergy to ",
        "allergies to ", "intolerant to ", "intolerance to ", "allergy: ", "allergies: ",
    )
    _RESTRICTION_SUFFIXES = (" allergy", " allergies", " allergic", " intolerance", " intolerant")
    _PREFERENCE_KEYS = ("meal_preference", "meal_source", "pantry_ingredients", "daily_budget")

    def __init__(self, username, name, age, gender, weight_kg, height_cm,
                 activity_level, goal, dietary_restrictions=None, preferences=None):
        self.__username = str(username).strip()
        self.__name = str(name).strip()
        self.__age = int(age)
        self.__gender = self.normalize_gender(gender)
        self.__weight_kg = float(weight_kg)
        self.__height_cm = float(height_cm)
        self.__activity_level = self.normalize_activity_level(activity_level)
        self.__goal = str(goal).strip().lower()
        self.__dietary_restrictions = self.normalize_restrictions(dietary_restrictions)
        self.__preferences = dict(preferences or {})

    # ---- read-only properties -------------------------------------------
    @property
    def username(self) -> str:
        return self.__username

    @property
    def name(self) -> str:
        return self.__name

    @property
    def age(self) -> int:
        return self.__age

    @property
    def gender(self) -> str:
        return self.__gender

    @property
    def weight_kg(self) -> float:
        return self.__weight_kg

    @property
    def height_cm(self) -> float:
        return self.__height_cm

    @property
    def activity_level(self) -> str:
        return self.__activity_level

    @property
    def goal(self) -> str:
        return self.__goal

    @property
    def dietary_restrictions(self) -> List[str]:
        return list(self.__dietary_restrictions)

    @property
    def preferences(self) -> Dict[str, Any]:
        """The most recent meal-plan preferences, remembered for next time."""
        return dict(self.__preferences)

    def remember_preferences(self, preferences: Dict[str, Any]) -> None:
        self.__preferences = {key: preferences[key] for key in self._PREFERENCE_KEYS if key in preferences}

    # ---- normalisation (shared by the CLI, the web form and stored data) --
    @staticmethod
    def normalize_gender(gender: Any) -> str:
        value = str(gender).strip().lower()
        return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(value, value)

    @classmethod
    def normalize_activity_level(cls, activity_level: Any) -> str:
        value = str(activity_level).strip().lower().replace(" ", "_")
        return cls._ACTIVITY_ALIASES.get(value, value)

    @classmethod
    def normalize_restrictions(cls, dietary_restrictions: Any) -> List[str]:
        """Turn free text such as "allergic to peanuts and dairy" into a
        list of individual restriction terms."""
        if isinstance(dietary_restrictions, str):
            restrictions = dietary_restrictions.replace(";", ",").split(",")
        else:
            restrictions = dietary_restrictions or []

        normalized = []
        for restriction in restrictions:
            value = restriction.strip().lower().replace(" and ", ",")
            for prefix in cls._RESTRICTION_PREFIXES:
                if value.startswith(prefix):
                    value = value[len(prefix):]
                    break
            for suffix in cls._RESTRICTION_SUFFIXES:
                if value.endswith(suffix):
                    value = value[:-len(suffix)]
                    break
            for item in value.split(","):
                item = item.strip(" \t\r\n.:;()")
                if item and item not in cls._NO_RESTRICTION_WORDS:
                    normalized.append(item)
        return normalized

    # ---- persistence -------------------------------------------------------
    def to_dict(self, include_preferences: bool = True) -> Dict[str, Any]:
        data = {
            "username": self.__username,
            "name": self.__name,
            "age": self.__age,
            "gender": self.__gender,
            "weight_kg": self.__weight_kg,
            "height_cm": self.__height_cm,
            "activity_level": self.__activity_level,
            "goal": self.__goal,
            "dietary_restrictions": list(self.__dietary_restrictions),
        }
        if include_preferences:
            data.update(self.__preferences)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        preferences = {key: data[key] for key in cls._PREFERENCE_KEYS if key in data}
        return cls(
            username=data["username"], name=data["name"], age=data["age"],
            gender=data["gender"], weight_kg=data["weight_kg"], height_cm=data["height_cm"],
            activity_level=data["activity_level"], goal=data["goal"],
            dietary_restrictions=data.get("dietary_restrictions"),
            preferences=preferences,
        )

    def updated_with(self, changes: Dict[str, Any]) -> "UserProfile":
        """A new profile with the given fields replaced (this one is unchanged)."""
        data = self.to_dict()
        data.update({key: value for key, value in changes.items() if value is not None})
        return UserProfile.from_dict(data)


class MealPlanRequest:
    """What the user wants for one particular day. Refers to a UserProfile
    (association): the profile outlives any single request."""

    SOURCE_ALIASES = {"in": "home_cooked", "home_cooked": "home_cooked",
                      "out": "eat_out", "eat_out": "eat_out"}
    SOURCE_LABELS = {"home_cooked": "in", "eat_out": "out", "both": "both"}

    def __init__(self, profile: UserProfile, meal_preference: str = "", meal_source: str = "both",
                 pantry_ingredients: Any = None, daily_budget: Optional[float] = None,
                 strict_budget: bool = False):
        self._profile = profile
        self._meal_preference = str(meal_preference or "").strip()
        self._meal_source = self.SOURCE_ALIASES.get(str(meal_source).strip().lower(), "both")
        if isinstance(pantry_ingredients, str):
            pantry_ingredients = [item.strip() for item in pantry_ingredients.split(",") if item.strip()]
        self._pantry_ingredients = list(pantry_ingredients or [])
        self._daily_budget = daily_budget
        self._strict_budget = strict_budget

    @property
    def profile(self) -> UserProfile:
        return self._profile

    @property
    def meal_preference(self) -> str:
        return self._meal_preference

    @property
    def meal_source(self) -> str:
        """Canonical value: ``home_cooked``, ``eat_out`` or ``both``."""
        return self._meal_source

    @property
    def meal_source_label(self) -> str:
        """The wording shown to the user and the AI: ``in``, ``out`` or ``both``."""
        return self.SOURCE_LABELS[self._meal_source]

    @property
    def pantry_ingredients(self) -> List[str]:
        return list(self._pantry_ingredients)

    @property
    def daily_budget(self) -> Optional[float]:
        return self._daily_budget

    def to_preferences(self) -> Dict[str, Any]:
        return {
            "meal_preference": self._meal_preference,
            "meal_source": self._meal_source,
            "pantry_ingredients": list(self._pantry_ingredients),
            "daily_budget": self._daily_budget,
        }

    @property
    def strict_budget(self) -> bool:
        """A request-only hard limit; existing budgets remain advisory by default."""
        return self._strict_budget


class NutritionTargets:
    """Calorie and macro targets worked out by NutritionCalculator."""

    def __init__(self, bmr: float, tdee: float, daily_calories: float, per_meal_calories: float,
                 daily_macros: Dict[str, float], per_meal_macros: Dict[str, float]):
        self._bmr = bmr
        self._tdee = tdee
        self._daily_calories = daily_calories
        self._per_meal_calories = per_meal_calories
        self._daily_macros = dict(daily_macros)
        self._per_meal_macros = dict(per_meal_macros)

    @property
    def bmr(self) -> float:
        return self._bmr

    @property
    def tdee(self) -> float:
        return self._tdee

    @property
    def daily_calories(self) -> float:
        return self._daily_calories

    @property
    def per_meal_calories(self) -> float:
        return self._per_meal_calories

    @property
    def daily_macros(self) -> Dict[str, float]:
        return dict(self._daily_macros)

    @property
    def per_meal_macros(self) -> Dict[str, float]:
        return dict(self._per_meal_macros)


class MealCandidate:
    """One recommendation returned by the AI, before filtering and ranking."""

    def __init__(self, meal_type, meal_name, source_type, calories, protein_g, carbs_g, fat_g,
                 estimated_cost, portion_size="", pantry_ingredients_used=None,
                 ingredients=None, recipe_steps=None):
        self._meal_type = meal_type
        self._meal_name = meal_name
        self._source_type = source_type
        self._calories = calories
        self._protein_g = protein_g
        self._carbs_g = carbs_g
        self._fat_g = fat_g
        self._estimated_cost = estimated_cost
        self._portion_size = portion_size
        self._pantry_ingredients_used = list(pantry_ingredients_used or [])
        self._ingredients = list(ingredients or [])
        self._recipe_steps = list(recipe_steps or [])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MealCandidate":
        return cls(
            meal_type=data.get("meal_type", ""), meal_name=data.get("meal_name", ""),
            source_type=data.get("meal_source_type", ""), calories=data.get("calories", 0),
            protein_g=data.get("protein_g", 0), carbs_g=data.get("carbs_g", 0),
            fat_g=data.get("fat_g", 0), estimated_cost=data.get("estimated_cost"),
            portion_size=data.get("portion_size", ""),
            pantry_ingredients_used=data.get("pantry_ingredients_used"),
            ingredients=data.get("ingredients"), recipe_steps=data.get("recipe_steps"),
        )

    @property
    def meal_type(self) -> str:
        return self._meal_type

    @property
    def slot(self) -> str:
        return str(self._meal_type).strip().lower()

    @property
    def meal_name(self) -> str:
        return self._meal_name

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def calories(self) -> float:
        return self._calories

    @property
    def protein_g(self) -> float:
        return self._protein_g

    @property
    def carbs_g(self) -> float:
        return self._carbs_g

    @property
    def fat_g(self) -> float:
        return self._fat_g

    @property
    def estimated_cost(self) -> Optional[float]:
        return self._estimated_cost

    @property
    def portion_size(self) -> str:
        return self._portion_size

    @property
    def ingredients(self) -> List[Any]:
        return list(self._ingredients)

    @property
    def ingredient_names(self) -> List[str]:
        return IngredientParser.names(self._ingredients)

    @property
    def recipe_steps(self) -> List[str]:
        return list(self._recipe_steps)


class PlannedMeal:
    """A chosen meal as shown to the user: tags rather than raw numbers, so
    the display is stable against small drift in the AI's estimates. The raw
    calories and cost are kept privately for the plan's totals."""

    def __init__(self, slot, name, source_type, calorie_range, calorie_tag, protein_tag,
                 carbs_tag, fat_tag, cost_tier, ingredients=None, calories=None, cost=None,
                 recipe_steps=None, portion_size=""):
        self._slot = slot
        self._name = name
        self._source_type = source_type
        self._calorie_range = calorie_range
        self._calorie_tag = calorie_tag
        self._protein_tag = protein_tag
        self._carbs_tag = carbs_tag
        self._fat_tag = fat_tag
        self._cost_tier = cost_tier
        self._ingredients = list(ingredients or [])
        self._calories = calories
        self._cost = cost
        self._recipe_steps = [str(step).strip() for step in recipe_steps or [] if str(step).strip()]
        self._portion_size = str(portion_size or "")

    @property
    def slot(self) -> str:
        return self._slot

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def calorie_tag(self) -> str:
        return self._calorie_tag

    @property
    def cost_tier(self) -> str:
        return self._cost_tier

    @property
    def ingredients(self) -> List[Any]:
        """Ingredients of a home-cooked meal; empty for eat-out meals."""
        return list(self._ingredients)

    @property
    def ingredient_names(self) -> List[str]:
        return IngredientParser.names(self._ingredients)

    @property
    def ingredient_labels(self) -> List[str]:
        return IngredientParser.labels(self._ingredients)

    @property
    def recipe_steps(self) -> List[str]:
        """Cooking instructions of a home-cooked meal; empty for eat-out meals."""
        return list(self._recipe_steps)

    @property
    def portion_size(self) -> str:
        return self._portion_size

    @property
    def calories(self) -> Optional[float]:
        return self._calories

    @property
    def cost(self) -> Optional[float]:
        return self._cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "source_type": self._source_type,
            "calorie_range": self._calorie_range,
            "calorie_tag": self._calorie_tag,
            "protein_tag": self._protein_tag,
            "carbs_tag": self._carbs_tag,
            "fat_tag": self._fat_tag,
            "cost_tier": self._cost_tier,
            "pantry_ingredients_used": list(self._ingredients),
            "recipe_steps": list(self._recipe_steps),
            "portion_size": self._portion_size,
        }

    @classmethod
    def from_dict(cls, slot: str, data: Dict[str, Any]) -> "PlannedMeal":
        """Plans saved before the recipe feature have no steps; they load with none."""
        return cls(
            slot=slot, name=data.get("name"), source_type=data.get("source_type"),
            calorie_range=data.get("calorie_range"), calorie_tag=data.get("calorie_tag"),
            protein_tag=data.get("protein_tag"), carbs_tag=data.get("carbs_tag"),
            fat_tag=data.get("fat_tag"), cost_tier=data.get("cost_tier"),
            ingredients=data.get("pantry_ingredients_used"),
            recipe_steps=data.get("recipe_steps"), portion_size=data.get("portion_size", ""),
        )


class Recipe:
    """NEW FEATURE: the cooking instructions for one home-cooked meal.

    A read-only value object built from a PlannedMeal by RecipeBuilder."""

    def __init__(self, slot: str, name: str, portion_size: str = "",
                 ingredients: Optional[List[str]] = None, steps: Optional[List[str]] = None):
        self._slot = slot
        self._name = name
        self._portion_size = portion_size
        self._ingredients = list(ingredients or [])
        self._steps = list(steps or [])

    @property
    def slot(self) -> str:
        return self._slot

    @property
    def name(self) -> str:
        return self._name

    @property
    def portion_size(self) -> str:
        return self._portion_size

    @property
    def ingredients(self) -> List[str]:
        return list(self._ingredients)

    @property
    def steps(self) -> List[str]:
        return list(self._steps)

    @property
    def has_steps(self) -> bool:
        return bool(self._steps)

    def to_dict(self) -> Dict[str, Any]:
        return {"slot": self._slot, "name": self._name, "portion_size": self._portion_size,
                "ingredients": list(self._ingredients), "steps": list(self._steps)}

    def to_text(self) -> str:
        lines = [f"{self._slot.capitalize()}: {self._name}"]
        if self._portion_size:
            lines.append(f"Serves: {self._portion_size}")
        if self._ingredients:
            lines.append("Ingredients:")
            lines.extend(f"  - {label}" for label in self._ingredients)
        if self._steps:
            lines.append("Steps:")
            lines.extend(f"  {number}. {step}" for number, step in enumerate(self._steps, start=1))
        else:
            lines.append("No cooking instructions were saved for this meal.")
        return "\n".join(lines)


class MealPlan:
    """A day's plan. It owns its PlannedMeal objects (composition): meals are
    created for the plan, belong to it, and are discarded with it."""

    SLOTS = ("breakfast", "lunch", "dinner")

    def __init__(self, daily_calorie_target: float, daily_budget: Optional[float] = None):
        self._daily_calorie_target = daily_calorie_target
        self._daily_budget = daily_budget
        self._meals: Dict[str, PlannedMeal] = {}
        self._missing_slots: List[str] = []
        self._total_calories = 0
        self._costs: List[float] = []
        self._stored_total_cost: Optional[float] = None
        self._ai_selection: Dict[str, Any] = {}

    def set_ai_selection(self, provider: str, score: float, components: Dict[str, float]) -> None:
        """Request-local comparison metadata; no database migration is needed."""
        self._ai_selection = {"provider": provider, "score": score, "components": dict(components)}

    @property
    def ai_selection(self) -> Dict[str, Any]:
        return dict(self._ai_selection)

    def add_meal(self, meal: PlannedMeal) -> None:
        self._meals[meal.slot] = meal
        self._total_calories += meal.calories or 0
        if meal.cost is not None:
            self._costs.append(meal.cost)

    def mark_missing(self, slot: str) -> None:
        self._missing_slots.append(slot)

    @property
    def meals(self) -> Dict[str, PlannedMeal]:
        return dict(self._meals)

    @property
    def missing_slots(self) -> List[str]:
        return list(self._missing_slots)

    @property
    def is_complete(self) -> bool:
        return not self._missing_slots

    @property
    def daily_calorie_target(self) -> float:
        return self._daily_calorie_target

    @property
    def daily_budget(self) -> Optional[float]:
        return self._daily_budget

    @property
    def total_calories(self) -> float:
        return self._total_calories

    @property
    def total_cost(self) -> Optional[float]:
        if self._stored_total_cost is not None:
            return self._stored_total_cost
        return round(sum(self._costs), 2) if self._costs else None

    @property
    def within_budget(self) -> bool:
        if self._daily_budget is None or self.total_cost is None:
            return True
        return self.total_cost <= self._daily_budget

    @property
    def explanation(self) -> str:
        parts = [f"Totals {self._total_calories:.0f} kcal against a "
                 f"{self._daily_calorie_target:.0f} kcal daily target"]
        if self._daily_budget is not None and self.total_cost is not None:
            status = "within" if self.within_budget else "over"
            parts.append(f"an estimated {self.total_cost:.2f} — {status} "
                         f"your {self._daily_budget:.2f} daily budget")
        return "; ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meals": {slot: meal.to_dict() for slot, meal in self._meals.items()},
            "is_complete": self.is_complete,
            "missing_slots": list(self._missing_slots),
            "daily_calorie_target": self._daily_calorie_target,
            "total_calories": self._total_calories,
            "daily_budget": self._daily_budget,
            "total_cost": self.total_cost,
            "within_budget": self.within_budget,
            "explanation": self.explanation,
            "ai_selection": self.ai_selection,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MealPlan":
        """Rebuild a stored plan (history); totals are taken as stored."""
        plan = cls(data.get("daily_calorie_target", 0), data.get("daily_budget"))
        for slot, meal_data in (data.get("meals") or {}).items():
            plan._meals[slot] = PlannedMeal.from_dict(slot, meal_data)
        plan._missing_slots = list(data.get("missing_slots") or [])
        plan._total_calories = data.get("total_calories", 0)
        plan._stored_total_cost = data.get("total_cost")
        return plan


class ShoppingList:
    """Ingredients still to buy for a plan, and those the pantry already covers."""

    def __init__(self, to_buy: Optional[List[str]] = None, in_pantry: Optional[List[str]] = None):
        self._to_buy = list(to_buy or [])
        self._in_pantry = list(in_pantry or [])

    @property
    def items(self) -> List[str]:
        return list(self._to_buy)

    @property
    def covered_by_pantry(self) -> List[str]:
        return list(self._in_pantry)

    @property
    def is_empty(self) -> bool:
        return not self._to_buy

    def to_text(self) -> str:
        if not self._to_buy:
            return "Nothing to buy: your pantry covers every home-cooked ingredient."
        lines = ["Shopping list:"] + [f"  - {item}" for item in self._to_buy]
        if self._in_pantry:
            lines.append("Already in your pantry: " + ", ".join(self._in_pantry))
        return "\n".join(lines)
