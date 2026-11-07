from abc import ABC, abstractmethod
from typing import List, Optional


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
    SAFE_EXCEPTIONS = ["coconut", "nutmeg", "butternut", "water chestnut", "peanut-free"]

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
