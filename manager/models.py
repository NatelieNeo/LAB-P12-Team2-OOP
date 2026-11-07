"""Domain model classes for Calora.

Attributes are private or protected by naming convention (``__x`` / ``_x``)
and exposed through read-only properties, so other classes cannot put an
object into an inconsistent state by assigning to its fields directly.
"""
from typing import Any, Dict, List, Optional


class IngredientParser:
    """Reads the AI's ingredient entries, each a plain string or an
    ``{"item": ..., "quantity": ..., "unit": ...}`` object."""

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
                 pantry_ingredients: Any = None, daily_budget: Optional[float] = None):
        self._profile = profile
        self._meal_preference = str(meal_preference or "").strip()
        self._meal_source = self.SOURCE_ALIASES.get(str(meal_source).strip().lower(), "both")
        if isinstance(pantry_ingredients, str):
            pantry_ingredients = [item.strip() for item in pantry_ingredients.split(",") if item.strip()]
        self._pantry_ingredients = list(pantry_ingredients or [])
        self._daily_budget = daily_budget

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