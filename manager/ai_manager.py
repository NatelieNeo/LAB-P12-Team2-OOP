"""AI Manager: builds the prompt, calls an AI provider, retries, and hands back candidates."""
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from config import GEMINI_MODEL, MAX_RETRIES, RETRY_DELAY_SECONDS
from manager.ai_handler_manager import AIHandlerManager
from manager.exceptions import AIConnectionError, AIResponseError
from manager.models import MealCandidate, MealPlanRequest, NutritionTargets

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Interface: anything that can turn a prompt into response text."""

    @property
    def name(self) -> str:
        """A label for logs and error messages."""
        return type(self).__name__

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's raw text. Raise AIConnectionError if it cannot be reached."""


class GeminiProvider(AIProvider):
    """Talks to Google Gemini. The API key is private and read from the environment."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.__api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self._model = model
        self._client = None
        # Several threads (parallel runs, background plan jobs) may share one provider,
        # so the lazy client creation is guarded by a lock.
        self._client_lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def _get_client(self):
        if not self.__api_key:
            raise AIConnectionError("GEMINI_API_KEY was not found. Add it to your .env file.", retryable=False)
        with self._client_lock:
            if self._client is None:
                try:
                    from google import genai
                    self._client = genai.Client(api_key=self.__api_key)
                except Exception as error:
                    raise AIConnectionError(
                        f"Failed to initialise the Gemini client: {error}", retryable=False) from error
            return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 1, "seed": 42},
            )
        except Exception as error:
            raise AIConnectionError(f"Gemini request failed: {error}") from error
        return response.text


class StubAIProvider(AIProvider):
    """Offline provider for tests and demos: replays canned responses in order
    (the last one repeats). An Exception instance in the list is raised instead."""

    def __init__(self, responses: Union[str, Exception, Sequence[Union[str, Exception]]]):
        single = isinstance(responses, (str, Exception))
        self._responses = [responses] if single else list(responses)
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    def generate(self, prompt: str) -> str:
        item = self._responses[min(self._calls, len(self._responses) - 1)]
        self._calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class PromptBuilder:
    """Builds the Gemini prompt from a request and the application-calculated targets."""

    TEMPLATE = """
You are the meal recommendation engine for Calora, a Singapore-based personalised meal planning application.

Your task is to analyse the user's profile, fitness goal, dietary requirements, food preferences and budget, then recommend the food options that best match their requirements.


USER PROFILE
# Basic information collected from io_manager.py

Name: {name}
Age: {age}
Gender: {gender}
Weight: {weight_kg} kg
Height: {height_cm} cm
Activity level: {activity_level}
Goal: {goal}


USER MEAL PREFERENCES
# Preferences and restrictions collected from io_manager.py

Dietary restrictions/allergies: {dietary_restrictions}
Meal preference: {meal_preference}
Meal source: {meal_source}
Available pantry ingredients: {pantry_ingredients}
Daily budget: SGD {daily_budget}


NUTRITION ESTIMATION
Use the application-calculated values below as the authoritative nutrition targets; do not calculate a different target.

Daily targets:
- calories: {daily_calories:.0f} kcal
- protein: {daily_protein:.1f} g
- carbohydrates: {daily_carbs:.1f} g
- fat: {daily_fat:.1f} g

For planning, use three meals per day. Each meal's calorie target is approximately {per_meal_calories:.0f} kcal, with per-meal macro targets of {meal_protein:.1f} g protein, {meal_carbs:.1f} g carbohydrates, and {meal_fat:.1f} g fat.

The estimated_nutrition_targets object in your JSON must repeat the daily targets above.


FOOD RECOMMENDATIONS
Generate exactly 10 food recommendations that best match the user's requirements.
Use only breakfast, lunch, or dinner as meal_type. Include at least 3 recommendations for each of these three meal types.
Aim for realistic serving sizes whose calories are within approximately 80%-120% of the per-meal calorie target above. Prefer options within that range rather than consistently undersized meals; do not inflate portions or nutrition estimates to hit a number.
The user's meal preference/craving is "{meal_preference}". Treat this as an important preference: include at least 2 recommendations matching it for each meal type whenever safe, practical options exist. Keep recommendations within the user's calorie and macro targets; do not ignore the craving just because another cuisine is marginally closer nutritionally. If a matching option would conflict with an allergy, dietary restriction, or realistic budget, exclude it and explain nothing outside the JSON.

Consider all of the following when generating meals:

- estimated nutritional requirements
- fitness goal
- dietary restrictions and allergies
- meal preference
- available pantry ingredients
- budget
- practicality and availability

Never recommend food containing anything listed under the user's dietary restrictions or allergies.


GENERAL RECOMMENDATION RULES
- AI handler validates format
- AI checks recommendations against restrictions
- Code removes incompatible or uncertain meals
- Display remaining meals


MEAL SOURCE
# Every recommendation must have a meal_source_type.

meal_source_type must be exactly one of:

- eat_out
- home_cooked

Follow the user's meal_source.

If meal_source is "in":
Only recommend home_cooked meals.

If meal_source is "out":
Only recommend eat_out meals.

If meal_source is "both":
Recommendations may contain both eat_out and home_cooked options.

# eat_out OPTIONS
Recommend realistic and affordable local food

For meals where meal_source_type is "eat_out":

- Prioritise food that is affordable and commonly available in Singapore.
- Prefer realistic hawker centre, coffee shop and food court options.
- Local dishes may include chicken rice, economical rice, yong tau foo, fishball noodles and other suitable Singapore dishes.
- Consider the user's fitness and nutritional goals.
- Give a realistic portion size.
- Estimate the calories for the portion.
- Estimate protein, carbohydrates and fat for the portion.
- Estimate the price in Singapore Dollars.
- Do not invent a specific restaurant, stall or business.
- In ingredients, list the known main ingredients in the dish, including sauces, stock, and common condiments when known. Ingredient names are enough; quantities are not needed.
- Never leave ingredients empty for an eat-out dish. If you cannot identify its ingredients reliably, do not recommend that dish.
- Do not claim a dish is safe for an allergy unless its ingredients are known.
- recipe_steps must be an empty list.

# home_cooked OPTIONS
# Include measurements and cooking instructions for recipes

For meals where meal_source_type is "home_cooked":

- Prioritise ingredients already available in the user's pantry.
- Additional ingredients may be included when necessary.
- Provide every ingredient required to make one serving.
- Give the exact quantity and unit for each ingredient.
- Provide simple cooking instructions in the correct order.
- Estimate calories, protein, carbohydrates and fat for one serving.
- Estimate the cost of one serving in SGD.
- Keep recipes realistic and reasonably easy to prepare.

PANTRY INGREDIENTS
# Shows which ingredients the user already owns

For each home-cooked recommendation, return pantry_ingredients_used.

pantry_ingredients_used must contain only ingredients that:
1. Appear in the user's available pantry ingredients.
2. Are actually used in the recommended recipe.

If no pantry ingredients are used, return an empty list.

For eat-out recommendations, return an empty list. The ingredients field still lists the dish's known ingredients for dietary-restriction filtering.

NUTRITION INFORMATION
# Raw nutrition estimates for each recommendation
For EVERY food recommendation provide:

- calories
- protein_g
- carbs_g
- fat_g

These must be numerical estimates.

Do not use descriptions such as "high protein" instead of numerical values.


COST INFORMATION
# Estimated cost of buying or preparing one serving

For EVERY food recommendation provide:
- estimated_cost

estimated_cost must be a numerical value in Singapore Dollars.

Consider the user's supplied budget when ranking recommendations.


OUTPUT FORMAT

-Return ONLY valid JSON.
-Do not include Markdown.
-Do not include explanations before or after the JSON.
-Do not add additional fields.
-The "recommendations" list must contain exactly 10 recommendation objects using the structure shown below.

Use exactly the following structure:
{{
    "estimated_nutrition_targets": {{
        "calorie_target": 0,
        "protein_target_g": 0,
        "carb_target_g": 0,
        "fat_target_g": 0
    }},

    "recommendations": [
        {{
            "meal_type": "",
            "meal_name": "",
            "meal_source_type": "",
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "estimated_cost": 0.00,
            "portion_size": "",
            "pantry_ingredients_used": [],
            "ingredients": [],
            "recipe_steps": []
        }}
    ]
}}
"""

    def build(self, request: MealPlanRequest, targets: NutritionTargets) -> str:
        profile = request.profile
        daily, meal = targets.daily_macros, targets.per_meal_macros
        return self.TEMPLATE.format(
            name=profile.name, age=profile.age, gender=profile.gender,
            weight_kg=profile.weight_kg, height_cm=profile.height_cm,
            activity_level=profile.activity_level, goal=profile.goal,
            dietary_restrictions=", ".join(profile.dietary_restrictions) or "none",
            meal_preference=request.meal_preference,
            meal_source=request.meal_source_label,
            pantry_ingredients=", ".join(request.pantry_ingredients) or "none",
            daily_budget=request.daily_budget if request.daily_budget is not None else "none",
            daily_calories=targets.daily_calories,
            daily_protein=daily["protein_g"], daily_carbs=daily["carbs_g"], daily_fat=daily["fat_g"],
            per_meal_calories=targets.per_meal_calories,
            meal_protein=meal["protein_g"], meal_carbs=meal["carbs_g"], meal_fat=meal["fat_g"],
        )


class AIManager:

    def __init__(self, provider: AIProvider, handler: Optional[AIHandlerManager] = None,
                 prompt_builder: Optional[PromptBuilder] = None,
                 max_retries: int = MAX_RETRIES, retry_delay: float = RETRY_DELAY_SECONDS):
        self._provider = provider
        self._handler = handler or AIHandlerManager()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # Raises AIConnectionError or AIResponseError once every attempt has failed.
    def get_recommendations(self, request: MealPlanRequest, targets: NutritionTargets) -> List[MealCandidate]:
        prompt = self._prompt_builder.build(request, targets)
        last_error: Exception = AIResponseError("No attempt was made.")
        for attempt in range(1, self._max_retries + 1):
            logger.info("Calling the AI provider - attempt %s/%s", attempt, self._max_retries)
            try:
                text = self._provider.generate(prompt)
                candidates = self._handler.parse_response(text)
                logger.info("AI response successfully validated.")
                return candidates
            except AIConnectionError as error:
                last_error = error
                logger.error("AI connection problem on attempt %s: %s", attempt, error)
                if not error.retryable:
                    raise
            except AIResponseError as error:
                last_error = error
                logger.warning("Invalid AI response on attempt %s: %s", attempt, error)
            if attempt < self._max_retries and self._retry_delay:
                time.sleep(self._retry_delay)
        logger.error("AI request failed after %s attempts.", self._max_retries)
        raise last_error
