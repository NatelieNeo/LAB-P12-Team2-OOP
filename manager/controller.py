"""CaloraController coordinates the AI, Logic and Data managers for both user
interfaces, and AppFactory assembles the real objects (the composition root)."""
import logging
from typing import Dict, Optional

import config
from manager.ai_manager import AIManager, AIProvider, GeminiProvider
from manager.data_manager import DataManager, SqliteDataManager
from manager.exceptions import NoSuitableMealsError
from manager.job_manager import PlanJob, PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import MealPlan, MealPlanRequest, ShoppingList, UserProfile

logger = logging.getLogger(__name__)


class CaloraController:
    """The managers are injected (aggregation): each exists independently and
    can be swapped, e.g. the offline stub provider or the SQLite data manager."""

    def __init__(self, ai: AIManager, logic: LogicManager, data: DataManager, jobs: PlanJobManager):
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
        candidates = self._ai.get_recommendations(request, targets)
        plan = self._logic.build_meal_plan(candidates, request)
        if plan is None:
            raise NoSuitableMealsError("No meals fit your restrictions and preferences. Try relaxing them.")
        self._data.update_profile(profile.username, request.to_preferences())
        self._data.save_plan(plan, profile.username)
        return plan

    def get_shopping_list(self, plan: MealPlan, request: MealPlanRequest) -> ShoppingList:
        return self._logic.build_shopping_list(plan, request.pantry_ingredients)

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
    def build_data_manager() -> DataManager:
        return SqliteDataManager(config.SQLITE_DB_FILE)

    @staticmethod
    def build_controller(provider: Optional[AIProvider] = None, data: Optional[DataManager] = None,
                         jobs: Optional[PlanJobManager] = None) -> CaloraController:
        return CaloraController(
            ai=AIManager(provider or GeminiProvider()),
            logic=LogicManager(),
            data=data or AppFactory.build_data_manager(),
            jobs=jobs or PlanJobManager(),
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

