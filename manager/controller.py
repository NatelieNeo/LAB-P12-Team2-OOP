"""CaloraController coordinates the AI, Logic and Data managers for both user
interfaces, and AppFactory assembles the real objects (the composition root)."""
import logging
from typing import Dict, List, Optional

import config
from manager.ai_handler_manager import AIHandlerManager
from manager.ai_manager import AIManager, AIProvider, GeminiProvider
from manager.ai_parallel import ParallelAIProvider
from manager.data_manager import DataManager, JsonDataManager, SqliteDataManager
from manager.exceptions import NoSuitableMealsError
from manager.job_manager import PlanJob, PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import MealPlan, MealPlanRequest, Recipe, ShoppingList, UserProfile

logger = logging.getLogger(__name__)

class AppFactory:
    """Builds the real, wired-up object graph."""

    @staticmethod
    def build_data_manager(backend: Optional[str] = None) -> DataManager:
        backend_name = (backend or config.get_storage_backend()).lower()
        if backend_name == "sqlite":
            return SqliteDataManager(config.SQLITE_DB_FILE)
        return JsonDataManager(config.USER_FILE, config.MEAL_PLAN_FILE)

    @staticmethod
    def build_ai_provider() -> AIProvider:
        """Two Gemini models asked in parallel (first valid answer wins), or just the
        primary model when parallel mode is off or both names are the same."""
        primary = GeminiProvider(model=config.GEMINI_MODEL)
        fallback_model = config.get_fallback_model()
        if not config.is_parallel_ai_enabled() or not fallback_model or fallback_model == config.GEMINI_MODEL:
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
            ai=AIManager(provider or AppFactory.build_ai_provider()),
            logic=LogicManager(),
            data=data or AppFactory.build_data_manager(),
            jobs=jobs or PlanJobManager(),
        )
