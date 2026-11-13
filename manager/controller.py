"""Coordinate the AI, Logic, Data and Job managers; assemble them in AppFactory."""
import logging
import os
from typing import Dict, List, Optional

import config
from manager.ai_handler_manager import AIHandlerManager
from manager.ai_manager import (AIManager, AIProvider, ClaudeProvider, GeminiProvider,
                                MultiAIRecommendationManager)
from manager.ai_parallel import ParallelAIProvider
from manager.data_manager import DataManager, JsonDataManager, SqliteDataManager
from manager.exceptions import InvalidInputError, NoSuitableMealsError
from manager.job_manager import PlanJob, PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import MealPlan, MealPlanRequest, Recipe, ShoppingList, UserProfile

logger = logging.getLogger(__name__)


class CaloraController:
    """The managers are injected (aggregation): each exists independently and
    can be swapped, e.g. the offline stub provider or the SQLite data manager."""

    def __init__(self, ai: AIManager | MultiAIRecommendationManager, logic: LogicManager,
                 data: DataManager, jobs: PlanJobManager):
        self._ai = ai
        self._logic = logic
        self._data = data
        self._jobs = jobs

    # ---- profiles ---------------------------------------------------------
    def get_profile(self, username: str) -> Optional[UserProfile]:
        return self._data.get_profile(username.strip())

    def register_profile(self, profile: UserProfile) -> None:
        self._data.save_profile(profile)

    def update_profile(self, username: str, profile: UserProfile) -> bool:
        """Store the profile fields (preferences are left alone)."""
        return self._data.update_profile(username, profile.to_dict(include_preferences=False))

    # ---- plans ---------------------------------------------------------------
    def create_meal_plan(self, request: MealPlanRequest) -> MealPlan:
        """Targets -> AI candidates -> filtered, ranked plan -> saved.

        Raises AIConnectionError / AIResponseError if the AI fails, and
        NoSuitableMealsError if every candidate is filtered out."""
        profile = request.profile
        targets = self._logic.calculate_targets(profile)
        if isinstance(self._ai, MultiAIRecommendationManager):
            responses = self._ai.get_response_sets(request, targets)
            selection = self._logic.select_response_sets(responses, request)
            plan = selection.plan
            plan.set_ai_selection(selection.response.provider, selection.score, selection.components)
        else:
            candidates = self._ai.get_recommendations(request, targets)
            plan = self._logic.build_meal_plan(candidates, request)
        if plan is None:
            raise NoSuitableMealsError("No meals fit your restrictions and preferences. Try relaxing them.")
        self._data.update_profile(profile.username, request.to_preferences())
        self._data.save_plan(plan, profile.username)
        return plan

    def get_shopping_list(self, plan: MealPlan, request: MealPlanRequest) -> ShoppingList:
        return self._logic.build_shopping_list(plan, request.pantry_ingredients)

    def get_recipes(self, plan: MealPlan) -> List[Recipe]:
        """Step-by-step recipes for the plan's home-cooked meals."""
        return self._logic.build_recipes(plan)

    def get_history(self, username: str) -> Dict[str, MealPlan]:
        return self._data.plans_for_user(username)

    # ---- background jobs -----------------------------------------------------
    def submit_plan_job(self, request: MealPlanRequest) -> str:
        return self._jobs.submit(lambda: self.create_meal_plan(request), owner=request.profile.username, context=request)

    def get_plan_job(self, job_id: str) -> Optional[PlanJob]:
        return self._jobs.get(job_id)


class AppFactory:
    """Builds the real, wired-up object graph."""

    @staticmethod
    def build_data_manager(backend: Optional[str] = None) -> DataManager:
        backend_name = (backend or config.get_storage_backend()).strip().lower()
        if backend_name == "sqlite":
            return SqliteDataManager(config.SQLITE_DB_FILE)
        if backend_name == "json":
            return JsonDataManager(config.USER_FILE, config.MEAL_PLAN_FILE)
        raise InvalidInputError("CALORA_STORAGE must be sqlite or json.")

    @staticmethod
    def build_ai_provider() -> AIProvider:
        """Two Gemini models asked in parallel (first valid answer wins), or just the
        primary model when parallel mode is off or both names are the same."""
        primary = GeminiProvider()
        fallback_model = config.get_fallback_model()
        if not config.is_parallel_ai_enabled() or not fallback_model or fallback_model == config.get_gemini_model():
            return primary
        return ParallelAIProvider(
            [primary, GeminiProvider(model=fallback_model)],
            validator=AIHandlerManager().is_valid,
            timeout=config.PARALLEL_TIMEOUT_SECONDS,
        )

    @staticmethod
    def build_controller(provider: Optional[AIProvider] = None, data: Optional[DataManager] = None,
                         jobs: Optional[PlanJobManager] = None) -> CaloraController:
        return CaloraController(
            ai=AIManager(provider) if provider is not None else AppFactory.build_ai_manager(),
            logic=LogicManager(),
            data=data or AppFactory.build_data_manager(),
            jobs=jobs or PlanJobManager(),
        )

    @staticmethod
    def build_ai_manager() -> AIManager | MultiAIRecommendationManager:
        """Build the requested mode, with an explicitly logged Claude backup in multi mode."""
        mode = config.get_ai_mode()
        logger.info("[MultiAI] Configured AI mode: %s", mode)
        if mode == "multi":
            gemini, claude = GeminiProvider(), ClaudeProvider()
            if not config.is_claude_backup_enabled():
                return MultiAIRecommendationManager([gemini, claude])
            recommended_timeout = (config.MAX_RETRIES * config.get_claude_timeout()
                                   + (config.MAX_RETRIES - 1) * config.RETRY_DELAY_SECONDS
                                   + config.get_ai_timeout() + 5)
            if config.get_ai_total_timeout() < recommended_timeout:
                logger.warning("[MultiAI] AI_TOTAL_TIMEOUT_SECONDS is shorter than %.0fs needed for "
                               "three full Claude attempts and one Gemini backup; the collection "
                               "deadline takes precedence", recommended_timeout)
            logger.info("[MultiAI] Claude has up to three attempts, then one Gemini backup attempt")
            return MultiAIRecommendationManager(
                [gemini, claude], fallback_provider=GeminiProvider(model=config.get_gemini_backup_model()),
                fallback_for=claude.name)
        if mode == "claude":
            return AIManager(ClaudeProvider())
        # Preserve the old two-Gemini race only when AI_MODE is unspecified.
        return AIManager(GeminiProvider() if os.getenv("AI_MODE", "").strip()
                         else AppFactory.build_ai_provider())

