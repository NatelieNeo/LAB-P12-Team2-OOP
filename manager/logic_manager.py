from copy import deepcopy
from dataclasses import dataclass
@dataclass(frozen=True)
class UserProfile:
    gender: str
    weight_kg: float
    height_cm: float
    age: int
    activity_level: str
    goal: str


@dataclass(frozen=True)
class NutritionTargets:
    bmr: float
    tdee: float
    daily_calorie_target: float
    _protein_g: float
    _carbs_g: float
    _fat_g: float

    def per_meal_calories(self):
        return self.daily_calorie_target / 3

    @property
    def per_meal_macro_targets(self):
        return {
            "protein_g": self._protein_g,
            "carbs_g": self._carbs_g,
            "fat_g": self._fat_g,
        }

    def to_dict(self):
        return {
            "bmr": self.bmr,
            "tdee": self.tdee,
            "daily_calorie_target": self.daily_calorie_target,
            "per_meal_calorie_target": self.per_meal_calories(),
            "per_meal_macro_targets": self.per_meal_macro_targets,
        }


@dataclass(frozen=True)
class MealCandidate:
    meal_name: str = ""
    meal_source_type: str = ""
    meal_type: str = ""
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    estimated_cost: float | None = None
    ingredients: list | None = None

    def ingredient_names(self):
        names = []

        for ingredient in self.ingredients or []:
            if isinstance(ingredient, dict):
                item = ingredient.get("item")
                if item:
                    names.append(str(item))
            elif isinstance(ingredient, str) and ingredient:
                names.append(ingredient)

        return names


class PlannedMeal:
    def __init__(self, name, source_type, tags, ingredients):
        self._name = name
        self._source_type = source_type
        self._tags = dict(tags)
        self._ingredients = deepcopy(ingredients)

    @property
    def ingredients(self):
        return deepcopy(self._ingredients)

    def to_dict(self):
        return {
            "name": self._name,
            "source_type": self._source_type,
            **self._tags,
            # Preserve the existing Part 1 output.
            "pantry_ingredients_used": (
                self.ingredients
                if self._source_type == "home_cooked"
                else []
            ),
        }