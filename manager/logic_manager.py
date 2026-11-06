from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from manager.models import (
    MealCandidate,
    MealPlan,
    MealPlanRequest,
    NutritionTargets,
    PlannedMeal,
    UserProfile,
)

class NutritionCalculator:
    """Calculate BMR, TDEE and daily/per-meal nutrition targets."""

    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }

    GOAL_ADJUSTMENTS = {
        "lose": -500,
        "maintain": 0,
        "gain": 300,
    }

    MACRO_SPLITS = {
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

    MEALS_PER_DAY = len(MealPlan.SLOTS)

    def bmr(self, profile: UserProfile) -> float:
        base = (
            10 * profile.weight_kg
            + 6.25 * profile.height_cm
            - 5 * profile.age
        )

        return base + 5 if profile.gender == "male" else base - 161

    def tdee(self, profile: UserProfile) -> float:
        multiplier = self.ACTIVITY_MULTIPLIERS.get(
            profile.activity_level,
            self.ACTIVITY_MULTIPLIERS["sedentary"],
        )

        return self.bmr(profile) * multiplier

    def daily_calorie_target(self, tdee: float, goal: str) -> float:
        return tdee + self.GOAL_ADJUSTMENTS.get(goal, 0)

    def per_meal_calorie_target(
        self,
        daily_calorie_target: float,
    ) -> float:
        return daily_calorie_target / self.MEALS_PER_DAY

    def macro_targets(
        self,
        calories: float,
        goal: str,
    ) -> Dict[str, float]:
        split = self.MACRO_SPLITS.get(
            goal,
            self.MACRO_SPLITS["maintain"],
        )

        return {
            "protein_g": round(
                calories * split["protein"] / 4, 1
            ),
            "carbs_g": round(
                calories * split["carbs"] / 4, 1
            ),
            "fat_g": round(
                calories * split["fat"] / 9, 1
            ),
        }

    def targets(self, profile: UserProfile) -> NutritionTargets:
        bmr = self.bmr(profile)
        tdee = self.tdee(profile)

        daily = self.daily_calorie_target(
            tdee,
            profile.goal,
        )

        per_meal = self.per_meal_calorie_target(daily)

        return NutritionTargets(
            bmr=bmr,
            tdee=tdee,
            daily_calories=daily,
            per_meal_calories=per_meal,
            daily_macros=self.macro_targets(
                daily,
                profile.goal,
            ),
            per_meal_macros=self.macro_targets(
                per_meal,
                profile.goal,
            ),
        )

    