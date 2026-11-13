"""Logic Manager: nutrition maths, safety/source rules, ranking and plan assembly."""
from abc import ABC, abstractmethod
from itertools import product
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from manager.models import (
    AIResponseSet, ResponseSelection,
    MealCandidate, MealPlan, MealPlanRequest, NutritionTargets,
    PlannedMeal, Recipe, ShoppingList, UserProfile,
)
import config
from manager.exceptions import NoSuitableMealsError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Nutrition maths
# ---------------------------------------------------------------------
class NutritionCalculator:
    """BMR (Mifflin-St Jeor) -> TDEE -> goal adjustment -> per-meal targets.
    This is the single source of truth for what a user should eat; the AI's
    own estimate is never used."""

    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9,
    }
    GOAL_ADJUSTMENTS = {"lose": -500, "maintain": 0, "gain": 300}
    # protein/carbs/fat as a fraction of calories; "gain" leans toward protein
    MACRO_SPLITS = {
        "lose": {"protein": 0.35, "carbs": 0.35, "fat": 0.30},
        "maintain": {"protein": 0.30, "carbs": 0.40, "fat": 0.30},
        "gain": {"protein": 0.35, "carbs": 0.40, "fat": 0.25},
    }
    MEALS_PER_DAY = len(MealPlan.SLOTS)

    def bmr(self, profile: UserProfile) -> float:
        base = (10 * profile.weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age)
        return base + 5 if profile.gender == "male" else base - 161

    def tdee(self, profile: UserProfile) -> float:
        multiplier = self.ACTIVITY_MULTIPLIERS.get(
            profile.activity_level, self.ACTIVITY_MULTIPLIERS["sedentary"])
        return self.bmr(profile) * multiplier

    def daily_calorie_target(self, tdee: float, goal: str) -> float:
        return tdee + self.GOAL_ADJUSTMENTS.get(goal, 0)

    def per_meal_calorie_target(self, daily_calorie_target: float) -> float:
        return daily_calorie_target / self.MEALS_PER_DAY

    def macro_targets(self, calories: float, goal: str) -> Dict[str, float]:
        """Grams of protein/carbs/fat for a calorie figure (4, 4 and 9 kcal per gram)."""
        split = self.MACRO_SPLITS.get(goal, self.MACRO_SPLITS["maintain"])
        return {
            "protein_g": round((calories * split["protein"]) / 4, 1),
            "carbs_g": round((calories * split["carbs"]) / 4, 1),
            "fat_g": round((calories * split["fat"]) / 9, 1),
        }

    def targets(self, profile: UserProfile) -> NutritionTargets:
        bmr = self.bmr(profile)
        tdee = self.tdee(profile)
        daily = self.daily_calorie_target(tdee, profile.goal)
        per_meal = self.per_meal_calorie_target(daily)
        return NutritionTargets(
            bmr=bmr, tdee=tdee, daily_calories=daily, per_meal_calories=per_meal,
            daily_macros=self.macro_targets(daily, profile.goal),
            per_meal_macros=self.macro_targets(per_meal, profile.goal),
        )

    
# ---------------------------------------------------------------------
# Business rules: interface + implementations
# ---------------------------------------------------------------------
class MealRule(ABC):
    """Interface: a rule that removes unsuitable candidates for one slot."""

    @abstractmethod
    def apply(self, candidates: List[MealCandidate], request: MealPlanRequest) -> List[MealCandidate]:
        """Return the candidates that pass this rule."""


class DietarySafetyRule(MealRule):
    """Drops any dish that may contain a restricted food.

    Layers: (1) a restriction is expanded to related words ("nuts" -> almond,
    cashew...), (2) ingredients and dish name are substring-matched, ignoring
    known safe words such as "coconut", (3) dishes known for hidden allergens
    (satay...) are dropped, (4) eat-out dishes with no ingredient list are
    dropped as uncertain."""

    RESTRICTION_KEYWORDS = {
        "vegetarian": ["chicken", "beef", "pork", "lamb", "mutton", "fish", "anchovy",
                       "prawn", "shrimp", "crab", "lobster", "gelatin", "belacan"],
        "vegan": ["chicken", "beef", "pork", "lamb", "mutton", "fish", "anchovy", "prawn",
                  "shrimp", "crab", "lobster", "gelatin", "belacan", "egg", "milk", "cheese",
                  "butter", "cream", "yogurt", "yoghurt", "ghee", "whey", "honey"],
        "peanut": ["peanut", "groundnut", "satay", "kacang"],
        "nuts": ["peanut", "almond", "cashew", "walnut", "hazelnut",
                 "pistachio", "pecan", "groundnut", "satay"],
        "tree nut": ["almond", "cashew", "walnut", "hazelnut", "pistachio", "pecan"],
        "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "yoghurt", "ghee", "whey"],
        "shellfish": ["prawn", "shrimp", "crab", "lobster", "cockle", "belacan", "oyster"],
        "gluten": ["wheat", "flour", "bread", "noodle", "pasta", "soy sauce", "barley"],
        "egg": ["egg", "mayonnaise"],
        "soy": ["soy", "tofu", "tempeh", "miso"],
        "fish": ["fish", "anchovy", "ikan bilis"],
    }
    RISKY_DISHES = {
        "peanut": ["satay", "rojak", "pad thai", "kueh pie tee"],
        "nuts": ["satay", "rojak", "pad thai"],
        "shellfish": ["laksa", "hokkien mee", "rojak", "har cheong"],
        "gluten": ["laksa", "mee goreng"],
    }
    SAFE_EXCEPTIONS = ["coconut", "nutmeg", "butternut", "water chestnut"]

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None,
                 risky_dishes: Optional[Dict[str, List[str]]] = None):
        self._keywords = keywords if keywords is not None else self.RESTRICTION_KEYWORDS
        self._risky_dishes = risky_dishes if risky_dishes is not None else self.RISKY_DISHES

    def _contains_term(self, text: str, term: str) -> bool:
        for safe in self.SAFE_EXCEPTIONS:
            text = text.replace(safe, "")
        return term in text

    @staticmethod
    def _variants(restriction: str) -> List[str]:
        """"peanuts" should also catch "peanut oil"."""
        if restriction == "nut":
            return ["nut", "nuts"]
        singular = restriction[:-1] if restriction.endswith("s") and len(restriction) > 3 else restriction
        return [restriction] if singular == restriction else [restriction, singular]

    def apply(self, candidates: List[MealCandidate], request: MealPlanRequest) -> List[MealCandidate]:
        restrictions = request.profile.dietary_restrictions
        if not restrictions:
            return list(candidates)

        banned_terms, risky_dishes = set(), set()
        for restriction in restrictions:
            for variant in self._variants(restriction):
                banned_terms.add(variant)
                banned_terms.update(self._keywords.get(variant, []))
                risky_dishes.update(self._risky_dishes.get(variant, []))

        safe = []
        for candidate in candidates:
            names = [name.lower() for name in candidate.ingredient_names]
            meal_name = str(candidate.meal_name).lower()
            if candidate.source_type == "eat_out" and not names:
                continue
            if any(dish in meal_name for dish in risky_dishes):
                continue
            haystack = names + [meal_name]
            if not any(self._contains_term(text, term) for term in banned_terms for text in haystack):
                safe.append(candidate)
        return safe


class SourcePreferenceRule(MealRule):
    """Honours "eat in only" / "eat out only". It is a soft preference: if it
    would leave a slot empty, the unfiltered candidates are kept so a safe dish
    of the other source still gets ranked."""

    def apply(self, candidates: List[MealCandidate], request: MealPlanRequest) -> List[MealCandidate]:
        if request.meal_source == "both":
            return list(candidates)
        preferred = [c for c in candidates if c.source_type == request.meal_source]
        return preferred if preferred else list(candidates)


# ---------------------------------------------------------------------
# Ranking criteria: interface + implementations (lower score = better)
# ---------------------------------------------------------------------
class RankingCriterion(ABC):
    """Interface: one factor in how candidates are ordered. The list order
    given to LogicManager is the priority order."""

    @abstractmethod
    def score(self, candidate: MealCandidate, targets: NutritionTargets, request: MealPlanRequest) -> float:
        """Lower is better."""


class CalorieFitCriterion(RankingCriterion):
    """Distance from the per-meal calorie target."""

    def score(self, candidate, targets, request) -> float:
        return abs(candidate.calories - targets.per_meal_calories)


class MacroFitCriterion(RankingCriterion):
    """Sum of gram differences from the per-meal protein/carbs/fat targets."""

    def score(self, candidate, targets, request) -> float:
        macros = targets.per_meal_macros
        return (abs(candidate.protein_g - macros.get("protein_g", 0))
                + abs(candidate.carbs_g - macros.get("carbs_g", 0))
                + abs(candidate.fat_g - macros.get("fat_g", 0)))


class PreferenceCriterion(RankingCriterion):
    """0 if the dish matches the user's craving or cuisine, otherwise 1."""

    CUISINE_KEYWORDS = {
        "chinese": ("chinese", "wonton", "dumpling", "congee", "chow mein", "fried rice",
                    "yong tau foo", "chicken rice", "char siu", "dim sum", "bao"),
        "japanese": ("japanese", "sushi", "sashimi", "teriyaki", "ramen", "udon", "soba",
                     "bento", "miso", "katsu", "onigiri", "tempura"),
        "western": ("western", "burger", "sandwich", "sub", "steak", "fish and chips",
                    "roast chicken", "mashed potato", "baked potato", "omelette"),
        "indian": ("indian", "biryani", "tandoori", "roti prata", "naan", "masala",
                   "chapati", "dosa", "dal"),
        "korean": ("korean", "kimchi", "bibimbap", "bulgogi", "tteokbokki", "kimbap"),
        "thai": ("thai", "pad thai", "tom yum", "green curry", "massaman", "larb"),
        "malay": ("malay", "nasi lemak", "satay", "rendang", "mee rebus", "laksa"),
        "italian": ("italian", "pizza", "pasta", "lasagna", "risotto", "carbonara"),
        "mexican": ("mexican", "taco", "burrito", "quesadilla", "enchilada"),
    }
    FILLER_WORDS = {"food", "foods", "dish", "dishes", "meal", "meals", "cuisine", "style"}

    def matches(self, candidate: MealCandidate, meal_preference: str) -> bool:
        preference = str(meal_preference or "").strip().lower()
        if not preference:
            return True
        meal_text = " ".join([str(candidate.meal_name), *candidate.ingredient_names]).lower()
        for cuisine, keywords in self.CUISINE_KEYWORDS.items():
            if cuisine in preference:
                return any(keyword in meal_text for keyword in keywords)
        terms = [t for t in preference.replace(",", " ").split() if t not in self.FILLER_WORDS]
        return not terms or any(term in meal_text for term in terms)

    def score(self, candidate, targets, request) -> float:
        return 0.0 if self.matches(candidate, request.meal_preference) else 1.0


class CostCriterion(RankingCriterion):
    """Cheaper is better; an unknown cost ranks last (it is not treated as free)."""

    def score(self, candidate, targets, request) -> float:
        cost = candidate.estimated_cost
        return float("inf") if cost is None else cost


# ---------------------------------------------------------------------
# Tagging and shopping list
# ---------------------------------------------------------------------
class MealTagger:
    """Turns a chosen candidate into a PlannedMeal of deterministic tags. Our
    code computes the tags from the AI's numbers; the AI never labels itself."""

    CALORIE_FIT_TOLERANCE = 0.20

    def __init__(self, tolerance: float = CALORIE_FIT_TOLERANCE):
        self._tolerance = tolerance

    def tag_calorie_range(self, calories: float) -> str:
        if calories < 400:
            return "Low (<400 kcal)"
        if calories <= 700:
            return "Moderate (400-700 kcal)"
        return "High (>700 kcal)"

    def tag_calorie_fit(self, calories: float, per_meal_target: float) -> str:
        if calories < per_meal_target * (1 - self._tolerance):
            return "under_target"
        if calories > per_meal_target * (1 + self._tolerance):
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

    def annotate(self, candidate: MealCandidate, targets: NutritionTargets) -> PlannedMeal:
        home_cooked = candidate.source_type == "home_cooked"
        return PlannedMeal(
            slot=candidate.slot,
            name=candidate.meal_name,
            source_type=candidate.source_type,
            calorie_range=self.tag_calorie_range(candidate.calories),
            calorie_tag=self.tag_calorie_fit(candidate.calories, targets.per_meal_calories),
            protein_tag=self.tag_protein(candidate.protein_g),
            carbs_tag=self.tag_carbs(candidate.carbs_g),
            fat_tag=self.tag_fat(candidate.fat_g),
            cost_tier=self.tag_cost(candidate.estimated_cost),
            ingredients=candidate.ingredients if home_cooked else [],
            calories=candidate.calories,
            cost=candidate.estimated_cost,
            recipe_steps=candidate.recipe_steps if home_cooked else [],
            portion_size=candidate.portion_size if home_cooked else "",
        )


class ShoppingListBuilder:
    """NEW FEATURE: works out what to buy for a plan's home-cooked meals,
    leaving out anything the user's pantry already covers."""

    def build(self, plan: MealPlan, pantry: Sequence[str]) -> ShoppingList:
        pantry_lower = [str(item).strip().lower() for item in pantry or [] if str(item).strip()]
        to_buy, in_pantry, seen = [], [], set()
        for meal in plan.meals.values():
            if meal.source_type != "home_cooked":
                continue
            for label in meal.ingredient_labels:
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                if any(item in key for item in pantry_lower):
                    in_pantry.append(label)
                else:
                    to_buy.append(label)
        return ShoppingList(to_buy, in_pantry)


class RecipeBuilder:
    """NEW FEATURE: turns the home-cooked meals of a plan into step-by-step
    recipes, in breakfast, lunch, dinner order. Eat-out meals have no recipe."""

    def build(self, plan: MealPlan) -> List[Recipe]:
        meals = plan.meals
        recipes = []
        for slot in MealPlan.SLOTS:
            meal = meals.get(slot)
            if meal is None or meal.source_type != "home_cooked":
                continue
            recipes.append(Recipe(
                slot=slot, name=meal.name, portion_size=meal.portion_size,
                ingredients=meal.ingredient_labels, steps=meal.recipe_steps,
            ))
        return recipes


# ---------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------
class LogicManager:
    """Builds a day's meal plan from the AI's candidates.

    The rules and ranking criteria are injected (aggregation): adding a new
    rule or criterion needs a new class, not a change to this one (OCP)."""

    def __init__(self, rules: Optional[List[MealRule]] = None,
                 criteria: Optional[List[RankingCriterion]] = None,
                 calculator: Optional[NutritionCalculator] = None,
                 tagger: Optional[MealTagger] = None,
                 shopping_builder: Optional[ShoppingListBuilder] = None,
                 recipe_builder: Optional[RecipeBuilder] = None):
        self._rules = rules if rules is not None else [DietarySafetyRule(), SourcePreferenceRule()]
        self._criteria = criteria if criteria is not None else [
            CalorieFitCriterion(), MacroFitCriterion(), PreferenceCriterion(), CostCriterion()]
        self._calculator = calculator or NutritionCalculator()
        self._tagger = tagger or MealTagger()
        self._shopping = shopping_builder or ShoppingListBuilder()
        self._recipes = recipe_builder or RecipeBuilder()

    def calculate_targets(self, profile: UserProfile) -> NutritionTargets:
        return self._calculator.targets(profile)

    def group_by_slot(self, candidates: List[MealCandidate]) -> Dict[str, List[MealCandidate]]:
        """Split the flat list into breakfast/lunch/dinner. Any other meal type
        (e.g. "snack") is dropped, not folded into a real slot."""
        grouped = {slot: [] for slot in MealPlan.SLOTS}
        for candidate in candidates or []:
            if candidate.slot in grouped:
                grouped[candidate.slot].append(candidate)
        return grouped

    def rank_slot(self, candidates: List[MealCandidate], targets: NutritionTargets,
                  request: MealPlanRequest) -> List[MealCandidate]:
        """Filter one slot's candidates through every rule, then sort them by the
        criteria in priority order (calorie fit, macro fit, preference, cost)."""
        pool = list(candidates)
        for rule in self._rules:
            pool = rule.apply(pool, request)
        return sorted(pool, key=lambda c: tuple(
            criterion.score(c, targets, request) for criterion in self._criteria))

    def build_meal_plan(self, candidates: List[MealCandidate], request: MealPlanRequest) -> Optional[MealPlan]:
        """One dish per slot. Returns None only if every slot ends up empty;
        otherwise a (possibly partial) plan that reports its missing slots."""
        targets = self.calculate_targets(request.profile)
        grouped = self.group_by_slot(candidates)
        plan = MealPlan(targets.daily_calories, request.daily_budget)
        if request.strict_budget and request.daily_budget is not None:
            ranked_slots = [self.rank_slot(grouped[slot], targets, request) for slot in MealPlan.SLOTS]
            feasible = [combo for combo in product(*ranked_slots)
                        if all(c.estimated_cost is not None for c in combo)
                        and round(sum(c.estimated_cost for c in combo), 2) <= request.daily_budget]
            if not feasible:
                return None
            # Same criterion priority as rank_slot, aggregated over a whole day.
            best = min(feasible, key=lambda combo: tuple(
                sum(criterion.score(c, targets, request) for c in combo) for criterion in self._criteria))
            for candidate in best:
                plan.add_meal(self._tagger.annotate(candidate, targets))
            return plan
        for slot in MealPlan.SLOTS:
            ranked = self.rank_slot(grouped[slot], targets, request)
            if ranked:
                plan.add_meal(self._tagger.annotate(ranked[0], targets))
            else:
                plan.mark_missing(slot)
        return plan if plan.meals else None

    def build_shopping_list(self, plan: MealPlan, pantry: Sequence[str]) -> ShoppingList:
        return self._shopping.build(plan, pantry)

    def build_recipes(self, plan: MealPlan) -> List[Recipe]:
        return self._recipes.build(plan)
    @staticmethod
    def _fit(actual: float, target: float) -> float:
        """1 at target, decreasing linearly to 0 at 100% relative error."""
        return max(0.0, 1.0 - abs(actual - target) / max(abs(target), 1.0))

    @staticmethod
    def _dish_key(candidate: MealCandidate) -> str:
        """Ignore casing, punctuation and word order to count obvious duplicates."""
        return " ".join(sorted(set(re.findall(r"[a-z]+", candidate.meal_name.lower()))))

    def score_response(self, response: AIResponseSet, request: MealPlanRequest) -> Optional[ResponseSelection]:
        """Reject hard violations, then score the complete set without mixing providers.

        Nutrition uses each candidate's *per-meal* target. Budget uses the three
        selected meals together. Existing meal ranking determines that day's plan.
        """
        candidates = list(response.candidates)
        if len(candidates) != config.RECOMMENDATION_COUNT:
            return None
        # A response with even one unsafe recommendation is ineligible, regardless
        # of its other scores. No ingredient list means safety is unknown.
        if any(not c.ingredient_names for c in candidates):
            return None
        if len(DietarySafetyRule().apply(candidates, request)) != len(candidates):
            return None
        if request.meal_source != "both" and any(c.source_type != request.meal_source for c in candidates):
            return None
        plan = self.build_meal_plan(candidates, request)
        if plan is None or not plan.is_complete:
            return None
        targets = self.calculate_targets(request.profile)
        macros = targets.per_meal_macros
        count = len(candidates)
        preference = PreferenceCriterion()
        home = [c for c in candidates if c.source_type == "home_cooked"]
        pantry = [p.strip().lower() for p in request.pantry_ingredients if p.strip()]
        pantry_fit = [sum(any(p in name.lower() for p in pantry) for name in c.ingredient_names)
                      / max(1, len(c.ingredient_names)) for c in home]
        # Treat missing costs as unknown, never as free, even for injected results.
        known_costs = all(c.estimated_cost is not None for c in candidates)
        budget_fit = 1.0
        if not known_costs:
            budget_fit = 0.0
        elif request.daily_budget is not None and plan.total_cost is not None:
            budget_fit = min(1.0, request.daily_budget / max(plan.total_cost, 0.01))
        components = {
            "calories": sum(self._fit(c.calories, targets.per_meal_calories) for c in candidates) / count,
            "macros": sum(sum(self._fit(getattr(c, field), target) for field, target in macros.items())
                          / len(macros) for c in candidates) / count,
            "budget": budget_fit,
            "preference": sum(preference.matches(c, request.meal_preference) for c in candidates) / count,
            "source": sum(request.meal_source == "both" or c.source_type == request.meal_source
                          for c in candidates) / count,
            "pantry": sum(pantry_fit) / len(pantry_fit) if home and pantry else 1.0,
            "variety": len({self._dish_key(c) for c in candidates}) / count,
            "practicality": sum(c.source_type == "eat_out" or bool(c.recipe_steps) for c in candidates) / count,
        }
        weights = config.RESPONSE_SCORE_WEIGHTS
        score = 100 * sum(weights[key] * value for key, value in components.items()) / sum(weights.values())
        return ResponseSelection(response, plan, score, components)

    def select_response_sets(self, responses: Sequence[AIResponseSet], request: MealPlanRequest) -> ResponseSelection:
        """Choose by score, then daily cost, then response content, then provider label.

        Completion order and latency never affect suitability. Provider name is
        consulted only when scores, cost and recommendation content are identical.
        """
        started = time.perf_counter()
        evaluated = []
        for response in responses:
            result = self.score_response(response, request)
            if result is None:
                logger.info("[LogicManager] %s rejected by hard constraints/completeness", response.provider)
            else:
                evaluated.append(result)
                logger.info("[LogicManager] %s score: %.4f components=%s",
                            response.provider, result.score, result.components)
        if not evaluated:
            raise NoSuitableMealsError("No complete AI response meets your dietary, meal-source and strict-budget requirements.")

        def order(result: ResponseSelection):
            content = tuple(sorted((self._dish_key(c), c.slot, c.source_type, c.calories,
                                    c.protein_g, c.carbs_g, c.fat_g, c.estimated_cost or 0,
                                    tuple(sorted(c.ingredient_names))) for c in result.response.candidates))
            cost = result.plan.total_cost
            return (-result.score, cost if cost is not None else float("inf"), content, result.response.provider)

        selected = min(evaluated, key=order)
        logger.info("[LogicManager] Selected provider: %s; score %.4f; comparison %.6f seconds",
                    selected.response.provider, selected.score, time.perf_counter() - started)
        return selected
