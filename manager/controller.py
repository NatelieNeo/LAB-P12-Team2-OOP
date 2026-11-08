# CaloraController coordinates the AI, Logic and Data managers for both user
# interfaces, and AppFactory assembles the real objects (the composition root).
import logging
from typing import Dict, Optional

import config
from manager.ai_manager import AIManager, AIProvider, GeminiProvider
from manager.data_manager import DataManager, JsonDataManager, SqliteDataManager
from manager.exceptions import NoSuitableMealsError
from manager.job_manager import PlanJob, PlanJobManager
from manager.logic_manager import LogicManager
from manager.models import MealPlan, MealPlanRequest, ShoppingList, UserProfile

logger = logging.getLogger(__name__)


# The managers are injected (aggregation): each exists independently and
# can be swapped, e.g. the offline stub provider or the SQLite data manager.
class CaloraController:

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

    # Store the profile fields (preferences are left alone).
    def update_profile(self, username: str, profile: UserProfile) -> bool:
        return self._data.update_profile(username, profile.to_dict(include_preferences=False))
