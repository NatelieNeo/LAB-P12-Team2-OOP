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
class NutritionCalculator:
    _ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    _GOAL_ADJUSTMENTS = {
        "lose": -500,
        "maintain": 0,
        "gain": 300,
    }

    _MACRO_SPLITS = {
        "lose": {
            "protein": 0.35,
            "carbs": 0.35,
            "fat": 0.30,
        },
        "maintain": {
            "protein": 0.30,
            "carbs": 0.40,
            "fat": 0.30,
        },
        "gain": {
            "protein": 0.35,
            "carbs": 0.40,
            "fat": 0.25,
        },
    }

    def bmr(self, profile):
        base = (
            10 * profile.weight_kg
            + 6.25 * profile.height_cm
            - 5 * profile.age
        )

        gender = str(profile.gender).strip().lower()

        return base + 5 if gender in {"male", "m"} else base - 161

    def tdee(self, profile):
        activity = (
            str(profile.activity_level)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        activity = {
            "moderately_active": "moderate",
        }.get(activity, activity)

        multiplier = self._ACTIVITY_MULTIPLIERS.get(
            activity,
            self._ACTIVITY_MULTIPLIERS["sedentary"],
        )

        return self.bmr(profile) * multiplier

    def macro_targets(self, calorie_amount, goal):
        split = self._MACRO_SPLITS.get(
            goal,
            self._MACRO_SPLITS["maintain"],
        )

        return {
            "protein_g": round(
                calorie_amount * split["protein"] / 4, 1
            ),
            "carbs_g": round(
                calorie_amount * split["carbs"] / 4, 1
            ),
            "fat_g": round(
                calorie_amount * split["fat"] / 9, 1
            ),
        }

    def targets(self, profile):
        bmr = self.bmr(profile)
        tdee = self.tdee(profile)

        daily_target = (
            tdee + self._GOAL_ADJUSTMENTS.get(profile.goal, 0)
        )

        macros = self.macro_targets(
            daily_target / 3,
            profile.goal,
        )

        return NutritionTargets(
            bmr=bmr,
            tdee=tdee,
            daily_calorie_target=daily_target,
            _protein_g=macros["protein_g"],
            _carbs_g=macros["carbs_g"],
            _fat_g=macros["fat_g"],
        )