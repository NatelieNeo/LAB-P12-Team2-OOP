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

class MealTagger:
    """Convert a selected candidate into a meal with calculated tags."""

    CALORIE_FIT_TOLERANCE = 0.20

    def __init__(
        self,
        tolerance: float = CALORIE_FIT_TOLERANCE,
    ):
        self._tolerance = tolerance

    def tag_calorie_range(self, calories: float) -> str:
        if calories < 400:
            return "Low (<400 kcal)"
        if calories <= 700:
            return "Moderate (400-700 kcal)"
        return "High (>700 kcal)"

    def tag_calorie_fit(
        self,
        calories: float,
        per_meal_target: float,
    ) -> str:
        lower = per_meal_target * (1 - self._tolerance)
        upper = per_meal_target * (1 + self._tolerance)

        if calories < lower:
            return "under_target"
        if calories > upper:
            return "over_target"
        return "on_target"

    def tag_protein(self, protein_g: float) -> str:
        if protein_g < 10:
            return "Low (<10g)"
        if protein_g <= 20:
            return "Moderate (10-20g)"
        if protein_g <= 30:
            return "High (>20-30g)"
        return "Very High (>30g)"

    def tag_carbs(self, carbs_g: float) -> str:
        if carbs_g < 15:
            return "Low (<15g)"
        if carbs_g <= 30:
            return "Moderate (15-30g)"
        if carbs_g <= 60:
            return "High (>30-60g)"
        return "Very High (>60g)"

    def tag_fat(self, fat_g: float) -> str:
        if fat_g < 5:
            return "Low (<5g)"
        if fat_g <= 10:
            return "Moderate (5-10g)"
        if fat_g <= 20:
            return "High (>10-20g)"
        return "Very High (>20g)"

    def tag_cost(self, estimated_cost: Optional[float]) -> str:
        if estimated_cost is None:
            return "Unknown"
        if estimated_cost < 5:
            return "Budget (<$5)"
        if estimated_cost < 10:
            return "Moderate (<$10)"
        if estimated_cost < 15:
            return "Moderate (<$15)"
        if estimated_cost < 20:
            return "High (<$20)"
        return "Premium (>$20)"

    def annotate(
        self,
        candidate: MealCandidate,
        targets: NutritionTargets,
    ) -> PlannedMeal:
        home_cooked = candidate.source_type == "home_cooked"

        return PlannedMeal(
            slot=candidate.slot,
            name=candidate.meal_name,
            source_type=candidate.source_type,
            calorie_range=self.tag_calorie_range(
                candidate.calories
            ),
            calorie_tag=self.tag_calorie_fit(
                candidate.calories,
                targets.per_meal_calories,
            ),
            protein_tag=self.tag_protein(
                candidate.protein_g
            ),
            carbs_tag=self.tag_carbs(
                candidate.carbs_g
            ),
            fat_tag=self.tag_fat(
                candidate.fat_g
            ),
            cost_tier=self.tag_cost(
                candidate.estimated_cost
            ),
            ingredients=(
                candidate.ingredients
                if home_cooked
                else []
            ),
            calories=candidate.calories,
            cost=candidate.estimated_cost,
        )
    