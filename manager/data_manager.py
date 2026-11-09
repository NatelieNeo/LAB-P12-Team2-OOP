"""Data Manager: stores profiles and meal plans.

DataManager is an abstract class: it defines what every storage backend must
do and supplies the behaviour they share. JsonDataManager and SqliteDataManager
are interchangeable (LSP), so the rest of the program never cares which is used."""
import json
import logging
import os
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict

from config import MEAL_PLAN_FILE, SQLITE_DB_FILE, USER_FILE

logger = logging.getLogger(__name__)


class DataManager(ABC):
    """Contract for storing user profiles and meal plans."""

    @abstractmethod
    def save_profile(self, profile: UserProfile) -> None:
        """Store a new profile."""

    @abstractmethod
    def find_profiles(self, key: str, value: Any) -> Dict[str, UserProfile]:
        """Return stored profiles whose field `key` equals `value`, keyed by record id."""

    @abstractmethod
    def update_profile(self, username: str, changes: Dict[str, Any]) -> bool:
        """Merge `changes` into the named user's record. False if no such user."""

    @abstractmethod
    def save_plan(self, plan: MealPlan, user_id: str) -> None:
        """Store a meal plan for a user."""

    @abstractmethod
    def plans_for_user(self, user_id: str) -> Dict[str, MealPlan]:
        """Return that user's stored plans, keyed by record id, oldest first."""

    def profile_exists(self, username: str) -> bool:
        """Shared behaviour built on the abstract find_profiles()."""
        return bool(self.find_profiles("username", username))

    def get_profile(self, username: str) -> "UserProfile | None":
        matches = self.find_profiles("username", username)
        return next(iter(matches.values()), None)
