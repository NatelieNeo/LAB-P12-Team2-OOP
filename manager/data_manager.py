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
from manager.exceptions import StorageError
from manager.models import MealPlan, UserProfile

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

class JsonDataManager(DataManager):
    """Stores records in two JSON files. A lock makes every read-modify-write
    atomic between threads, so concurrent saves cannot overwrite each other."""

    def __init__(self, user_file: str = USER_FILE, plan_file: str = MEAL_PLAN_FILE):
        self._user_file = user_file
        self._plan_file = plan_file
        self._lock = threading.Lock()

    def _read_json(self, path: str) -> Dict[str, Any]:
        """A missing file is empty. A corrupt file is moved aside (not overwritten)
        and treated as empty."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            backup = path + ".corrupt"
            logger.warning("%s is not valid JSON; keeping a copy at %s", path, backup)
            try:
                os.replace(path, backup)
            except OSError as error:
                raise StorageError(f"Could not set aside corrupt file {path}: {error}") from error
            return {}
        except OSError as error:
            raise StorageError(f"Could not read {path}: {error}") from error

    def _write_json(self, path: str, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
            os.replace(temp_path, path)
        except OSError as error:
            raise StorageError(f"Could not write {path}: {error}") from error

    def save_profile(self, profile: UserProfile) -> None:
        with self._lock:
            users = self._read_json(self._user_file)
            users[str(uuid.uuid4())] = profile.to_dict()
            self._write_json(self._user_file, users)

    def find_profiles(self, key: str, value: Any) -> Dict[str, UserProfile]:
        with self._lock:
            users = self._read_json(self._user_file)
        return {entry_id: UserProfile.from_dict(data) for entry_id, data in users.items()
                if data.get(key) == value}

    def update_profile(self, username: str, changes: Dict[str, Any]) -> bool:
        with self._lock:
            users = self._read_json(self._user_file)
            for data in users.values():
                if data.get("username") == username:
                    data.update(changes)
                    self._write_json(self._user_file, users)
                    return True
        return False

    def save_plan(self, plan: MealPlan, user_id: str) -> None:
        record = plan.to_dict()
        record["user_id"] = user_id
        with self._lock:
            plans = self._read_json(self._plan_file)
            plans[str(uuid.uuid4())] = record
            self._write_json(self._plan_file, plans)

    def plans_for_user(self, user_id: str) -> Dict[str, MealPlan]:
        with self._lock:
            plans = self._read_json(self._plan_file)
        return {entry_id: MealPlan.from_dict(data) for entry_id, data in plans.items()
                if data.get("user_id") == user_id}