"""Optional no-cost demonstration of the real multi-AI collection path.

Run: python benchmark_multi_ai.py --delay 2
This module never creates an SDK client or sends a network request.
"""
import argparse
import json
import logging
import time

from manager.ai_manager import AIManager, AIProvider, MultiAIRecommendationManager, PromptBuilder
from manager.logic_manager import LogicManager, NutritionCalculator
from manager.models import MealPlanRequest, UserProfile


class DelayedProvider(AIProvider):
    """An injected provider with simulated I/O latency and a valid canned answer."""

    def __init__(self, name: str, response: str, delay: float):
        self._name, self._response, self._delay = name, response, delay

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> str:
        time.sleep(self._delay)
        return self._response


def run_benchmark(delay: float = 2.0) -> dict[str, float]:
    """Measure equivalent sequential/concurrent work with no paid API traffic."""
    if delay <= 0:
        raise ValueError("Use a positive simulated delay.")
    profile = UserProfile("demo", "Demo", 25, "female", 60, 165, "moderate", "maintain")
    request = MealPlanRequest(profile, "rice", "both", ["rice"], 15)
    targets = NutritionCalculator().targets(profile)
    meals = []
    for i, slot in enumerate(("breakfast",) * 3 + ("lunch",) * 4 + ("dinner",) * 3):
        meals.append({"meal_type": slot, "meal_name": f"Rice bowl {chr(65 + i)}",
                      "meal_source_type": "home_cooked", "calories": targets.per_meal_calories,
                      **targets.per_meal_macros, "estimated_cost": 3, "portion_size": "1 serving",
                      "pantry_ingredients_used": ["rice"], "ingredients": ["rice"],
                      "recipe_steps": ["Cook rice for 15 minutes."]})
    text = json.dumps({"estimated_nutrition_targets": {
        "calorie_target": targets.daily_calories, "protein_target_g": targets.daily_macros["protein_g"],
        "carb_target_g": targets.daily_macros["carbs_g"], "fat_target_g": targets.daily_macros["fat_g"]},
        "recommendations": meals})
    providers = [DelayedProvider("Simulated Gemini", text, delay), DelayedProvider("Simulated Claude", text, delay)]
    prompt = PromptBuilder().build(request, targets)
    started = time.perf_counter()
    for provider in providers:
        AIManager(provider)._request(prompt)
    sequential = time.perf_counter() - started
    started = time.perf_counter()
    responses = MultiAIRecommendationManager(providers, timeout=delay * 2 + 10).get_response_sets(request, targets)
    concurrent = time.perf_counter() - started
    selection = LogicManager().select_response_sets(responses, request)
    intervals_overlap = max(r.started_at for r in responses) < min(r.completed_at for r in responses)
    print(f"Simulated latency per provider: {delay:.3f}s (no API calls)")
    print(f"Sequential: {sequential:.3f}s; concurrent: {concurrent:.3f}s; speedup: {sequential / concurrent:.2f}x")
    print(f"Worker intervals overlap: {intervals_overlap}; selected: {selection.response.provider}; score: {selection.score:.2f}")
    return {"sequential": sequential, "concurrent": concurrent, "speedup": sequential / concurrent}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    run_benchmark(args.delay)
