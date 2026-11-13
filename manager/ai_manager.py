"""AI Manager: builds the prompt, calls an AI provider, retries, and hands back candidates."""
import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import config
from config import MAX_RETRIES, RETRY_DELAY_SECONDS
from manager.ai_handler_manager import AIHandlerManager
from manager.exceptions import AIConnectionError, AIResponseError, AllProvidersFailedError
from manager.models import AIResponseSet, MealCandidate, MealPlanRequest, NutritionTargets

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

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 client=None, timeout: Optional[float] = None):
        self.__api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self._model = model or config.get_gemini_model()
        self._client = client
        self._timeout = timeout if timeout is not None else config.get_ai_timeout()
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
            if self._client is not None:
                return self._client
            try:
                from google import genai
            except ImportError:
                raise AIConnectionError("Install google-genai to use Gemini.", retryable=False) from None
            # A fresh client belongs to this call/thread and is closed in generate().
            return genai.Client(api_key=self.__api_key, http_options={
                "timeout": int(self._timeout * 1000), "retry_options": {"attempts": 1}})

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        from google.genai import errors
        import httpx2
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={"response_mime_type": "application/json", "temperature": 1, "seed": 42,
                        "max_output_tokens": config.get_max_output_tokens()},
            )
        except (errors.APIError, httpx2.TransportError) as error:
            status = getattr(error, "code", None)
            raise AIConnectionError(
                f"Gemini request failed ({type(error).__name__}, status {status or 'network'}).",
                retryable=status is None or status in (408, 409, 429) or status >= 500) from None
        finally:
            if self._client is None:
                client.close()
        return response.text


class ClaudeProvider(AIProvider):
    """Official Anthropic Messages adapter; only text blocks enter JSON validation."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 client=None, timeout: Optional[float] = None):
        self.__api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        self._model = model or config.get_claude_model()
        self._client = client
        self._timeout = timeout if timeout is not None else config.get_ai_timeout()

    @property
    def name(self) -> str:
        return f"claude:{self._model}"

    def generate(self, prompt: str) -> str:
        """Make one SDK request; AIManager owns the bounded retry policy."""
        if not self.__api_key:
            raise AIConnectionError("ANTHROPIC_API_KEY was not found. Configure your environment.",
                                    retryable=False)
        try:
            import anthropic
        except ImportError:
            raise AIConnectionError("Install anthropic to use Claude.", retryable=False) from None
        client = self._client or anthropic.Anthropic(
            api_key=self.__api_key, timeout=self._timeout, max_retries=0)
        try:
            response = client.messages.create(
                model=self._model, max_tokens=config.get_max_output_tokens(),
                messages=[{"role": "user", "content": prompt}],
                system="Return only the JSON object requested by the user. Do not use Markdown.")
            if response.stop_reason == "max_tokens":
                raise AIResponseError("Claude reached the output-token limit; increase AI_MAX_OUTPUT_TOKENS.")
            return "".join(block.text for block in response.content if block.type == "text")
        except anthropic.APIError as error:
            status = getattr(error, "status_code", None)
            raise AIConnectionError(
                f"Claude request failed ({type(error).__name__}, status {status or 'network'}).",
                retryable=status is None or status in (408, 409, 429) or status >= 500) from None
        finally:
            if self._client is None:
                client.close()


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
    """Single business prompt shared by every provider."""

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
Strict daily budget: {strict_budget} (if true, the chosen breakfast, lunch and dinner must fit together).


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
Generate exactly {recommendation_count} food recommendations that best match the user's requirements.
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
- Put the cooking instructions in recipe_steps: a list of short strings, one action per step, in the correct order, with quantities and times where they matter. Do not number the steps.
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
-The "recommendations" list must contain exactly {recommendation_count} recommendation objects using the structure shown below.

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
            strict_budget=request.strict_budget,
            recommendation_count=config.RECOMMENDATION_COUNT,
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
        if max_retries < 1 or retry_delay < 0:
            raise ValueError("Use at least one attempt and a non-negative retry delay.")
        self._handler = handler or AIHandlerManager()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # Raises AIConnectionError or AIResponseError once every attempt has failed.
    def get_recommendations(self, request: MealPlanRequest, targets: NutritionTargets) -> List[MealCandidate]:
        prompt = self._prompt_builder.build(request, targets)
        return self._request(prompt)

    def _request(self, prompt: str, deadline: Optional[float] = None) -> List[MealCandidate]:
        """Retry only this provider; never replay a successful peer's request."""
        last_error: Exception = AIResponseError("No attempt was made.")
        for attempt in range(1, self._max_retries + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise AIConnectionError("AI collection deadline reached.")
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
                delay = self._retry_delay
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.perf_counter()))
                time.sleep(delay)
        logger.error("AI request failed after %s attempts.", self._max_retries)
        raise last_error


class MultiAIRecommendationManager:
    """Collect two independently validated sets; LogicManager chooses their quality.

    All per-request state is local. Each worker uses the existing AIManager retry
    and parsing path. A collection deadline bounds the caller's waiting time.
    """

    def __init__(self, providers: Sequence[AIProvider], handler: Optional[AIHandlerManager] = None,
                 prompt_builder: Optional[PromptBuilder] = None, timeout: Optional[float] = None,
                 max_retries: int = MAX_RETRIES, retry_delay: float = RETRY_DELAY_SECONDS):
        if len(providers) != 2 or len({p.name for p in providers}) != 2:
            raise ValueError("Multi-AI requires exactly two distinctly named providers.")
        self._providers = tuple(providers)
        self._handler = handler or AIHandlerManager()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._timeout = timeout if timeout is not None else config.get_ai_total_timeout()
        if not math.isfinite(self._timeout) or self._timeout <= 0 or max_retries < 1:
            raise ValueError("Timeout and maximum attempts must be positive.")
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def _ask(self, provider: AIProvider, prompt: str, deadline: float) -> AIResponseSet:
        started = time.perf_counter()
        logger.info("[%s] Thread started at %.6f", provider.name, started)
        try:
            candidates = AIManager(provider, handler=self._handler, max_retries=self._max_retries,
                                   retry_delay=self._retry_delay)._request(prompt, deadline)
            return AIResponseSet(provider.name, tuple(candidates), started, time.perf_counter())
        finally:
            completed = time.perf_counter()
            logger.info("[%s] Thread completed at %.6f (%.3f seconds)",
                        provider.name, completed, completed - started)

    def get_response_sets(self, request: MealPlanRequest, targets: NutritionTargets) -> List[AIResponseSet]:
        """Submit both calls before waiting and retain all valid completed answers."""
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        prompt = self._prompt_builder.build(request, targets)
        started = time.perf_counter()
        deadline = started + self._timeout
        responses, failures = [], {}
        logger.info("[MultiAI] Starting concurrent recommendation generation")
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="calora-ai")
        futures = {}

        def collect_future(future):
            provider = futures[future]
            try:
                result = future.result()
            except (AIConnectionError, AIResponseError) as error:
                failures[provider.name] = error
                logger.warning("[MultiAI] %s failed: %s", provider.name, error)
            else:
                if result.completed_at <= deadline:
                    responses.append(result)
                else:
                    failures[provider.name] = AIConnectionError("Collection timeout.")

        try:
            for provider in self._providers:
                futures[executor.submit(self._ask, provider, prompt, deadline)] = provider
            pending = set(futures)
            while pending:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                for future in done:
                    collect_future(future)
                if not done:
                    break
            for future in pending:
                if future.done():
                    collect_future(future)
                else:
                    provider = futures[future]
                    failures[provider.name] = AIConnectionError("Collection timeout.")
                    logger.warning("[MultiAI] %s reached collection timeout", provider.name)
        finally:
            # A running Python thread cannot be killed. SDK timeouts bound its
            # network call; cancellation prevents any queued work from starting.
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("[MultiAI] Collection completed in %.3f seconds (%s valid responses)",
                        time.perf_counter() - started, len(responses))
        if responses:
            return responses
        if failures and all(isinstance(e, AIResponseError) for e in failures.values()):
            raise AIResponseError("Neither AI provider returned valid meal recommendations.")
        raise AllProvidersFailedError("No AI provider succeeded. Check credentials, quota and connectivity.", failures)
