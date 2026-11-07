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
