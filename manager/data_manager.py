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
from manager.models import MealPlan, PlannedMeal, UserProfile

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

class SqliteDataManager(DataManager):
    """SQLite-backed storage that follows the app's relational schema.

    Profiles are stored in the normalized tables used by the SQL schema, not as
    JSON blobs, so CRUD operations map directly to users, restrictions,
    pantry ingredients, meal plans and plan meals.
    """

    _USER_COLUMNS = (
        "username", "name", "age", "gender", "weight_kg", "height_cm",
        "activity_level", "goal", "meal_preference", "meal_source", "daily_budget",
    )

    def __init__(self, db_path: str = SQLITE_DB_FILE):
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            connection = sqlite3.connect(self._db_path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = OFF")
            return connection
        except (sqlite3.Error, OSError) as error:
            raise StorageError(f"Could not open database {self._db_path}: {error}") from error

    def _execute(self, sql: str, params: tuple = (), fetch: bool = False):
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(sql, params)
                return cursor.fetchall() if fetch else cursor.rowcount
        except sqlite3.Error as error:
            raise StorageError(f"Database error: {error}") from error
        finally:
            connection.close()

    def _init_schema(self) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = OFF;

                    CREATE TABLE IF NOT EXISTS users (
                        user_id         TEXT PRIMARY KEY,
                        username        TEXT NOT NULL UNIQUE,
                        name            TEXT NOT NULL,
                        age             INTEGER CHECK (age > 0),
                        gender          TEXT CHECK (gender IN ('male', 'female', 'other')),
                        weight_kg       REAL CHECK (weight_kg > 0),
                        height_cm       REAL CHECK (height_cm > 0),
                        activity_level  TEXT,
                        goal            TEXT,
                        meal_preference TEXT,
                        meal_source     TEXT CHECK (meal_source IN ('home_cooked', 'eat_out', 'both')),
                        daily_budget    REAL CHECK (daily_budget >= 0)
                    );

                    CREATE TABLE IF NOT EXISTS user_dietary_restrictions (
                        user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        restriction TEXT NOT NULL,
                        PRIMARY KEY (user_id, restriction)
                    );

                    CREATE TABLE IF NOT EXISTS user_pantry_ingredients (
                        user_id    TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        ingredient TEXT NOT NULL,
                        PRIMARY KEY (user_id, ingredient)
                    );

                    CREATE TABLE IF NOT EXISTS meal_plans (
                        plan_id              TEXT PRIMARY KEY,
                        username             TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                        is_complete          INTEGER NOT NULL DEFAULT 0 CHECK (is_complete IN (0, 1)),
                        daily_calorie_target REAL,
                        total_calories       INTEGER,
                        daily_budget         REAL,
                        total_cost           REAL,
                        within_budget        INTEGER CHECK (within_budget IN (0, 1)),
                        explanation          TEXT
                    );

                    CREATE TABLE IF NOT EXISTS meal_plan_missing_slots (
                        plan_id TEXT NOT NULL REFERENCES meal_plans(plan_id) ON DELETE CASCADE,
                        slot    TEXT NOT NULL CHECK (slot IN ('breakfast', 'lunch', 'dinner')),
                        PRIMARY KEY (plan_id, slot)
                    );

                    CREATE TABLE IF NOT EXISTS meals (
                        meal_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                        plan_id       TEXT NOT NULL REFERENCES meal_plans(plan_id) ON DELETE CASCADE,
                        slot          TEXT NOT NULL CHECK (slot IN ('breakfast', 'lunch', 'dinner')),
                        name          TEXT NOT NULL,
                        source_type   TEXT CHECK (source_type IN ('home_cooked', 'eat_out')),
                        calorie_range TEXT CHECK (calorie_range IN (
                                              'Low (<400 kcal)', 'Moderate (400-700 kcal)', 'High (>700 kcal)')),
                        calorie_tag   TEXT CHECK (calorie_tag IN ('under_target', 'on_target', 'over_target')),
                        protein_tag   TEXT CHECK (protein_tag IN (
                                              'Low (<10g)', 'Moderate (10-20g)', 'High (>20-30g)', 'Very High (>30g)')),
                        carbs_tag     TEXT CHECK (carbs_tag IN (
                                              'Low (<15g)', 'Moderate (15-30g)', 'High (>30-60g)', 'Very High (>60g)')),
                        fat_tag       TEXT CHECK (fat_tag IN (
                                              'Low (<5g)', 'Moderate (5-10g)', 'High (>10-20g)', 'Very High (>20g)')),
                        cost_tier     TEXT NOT NULL DEFAULT 'Unknown' CHECK (cost_tier IN (
                                              'Unknown', 'Budget (<$5)', 'Moderate (<$10)', 'Moderate (<$15)',
                                              'High (<$20)', 'Premium (>$20)')),
                        UNIQUE (plan_id, slot)
                    );

                    CREATE TABLE IF NOT EXISTS meal_ingredients (
                        meal_id    INTEGER NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE,
                        ingredient TEXT NOT NULL,
                        PRIMARY KEY (meal_id, ingredient)
                    );

                    -- Recipe feature: portion size and ordered cooking steps of a home-cooked meal.
                    -- New tables (not new columns), so an existing calora.db upgrades in place.
                    CREATE TABLE IF NOT EXISTS meal_recipes (
                        meal_id      INTEGER PRIMARY KEY REFERENCES meals(meal_id) ON DELETE CASCADE,
                        portion_size TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS meal_recipe_steps (
                        meal_id     INTEGER NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE,
                        step_no     INTEGER NOT NULL CHECK (step_no > 0),
                        instruction TEXT NOT NULL,
                        PRIMARY KEY (meal_id, step_no)
                    );
                    """
                )
        except sqlite3.Error as error:
            raise StorageError(f"Could not prepare database: {error}") from error
        finally:
            connection.close()

    def _replace_array_values(self, table_name: str, foreign_id: str, column_name: str,
                             values: Any) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(f"DELETE FROM {table_name} WHERE user_id = ?", (foreign_id,))
                for value in values or []:
                    connection.execute(
                        f"INSERT INTO {table_name} (user_id, {column_name}) VALUES (?, ?)",
                        (foreign_id, str(value)),
                    )
        except sqlite3.Error as error:
            raise StorageError(f"Could not update {table_name}: {error}") from error
        finally:
            connection.close()

    def _user_id_for_username(self, username: str) -> str | None:
        row = self._execute(
            "SELECT user_id FROM users WHERE username = ?",
            (username,),
            fetch=True,
        )
        return row[0][0] if row else None

    def _load_user_record(self, user_id: str) -> Dict[str, Any]:
        row = self._execute(
            "SELECT user_id, username, name, age, gender, weight_kg, height_cm, "
            "activity_level, goal, meal_preference, meal_source, daily_budget "
            "FROM users WHERE user_id = ?",
            (user_id,),
            fetch=True,
        )
        if not row:
            return {}
        data = dict(row[0])
        data["dietary_restrictions"] = [
            item[0] for item in self._execute(
                "SELECT restriction FROM user_dietary_restrictions WHERE user_id = ? ORDER BY rowid",
                (user_id,),
                fetch=True,
            )
        ]
        data["pantry_ingredients"] = [
            item[0] for item in self._execute(
                "SELECT ingredient FROM user_pantry_ingredients WHERE user_id = ? ORDER BY rowid",
                (user_id,),
                fetch=True,
            )
        ]
        return data

    def save_profile(self, profile: UserProfile) -> None:
        user_id = str(uuid.uuid4())
        payload = profile.to_dict()
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO users (user_id, username, name, age, gender, weight_kg, height_cm, "
                    "activity_level, goal, meal_preference, meal_source, daily_budget) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        profile.username,
                        payload["name"],
                        payload["age"],
                        payload["gender"],
                        payload["weight_kg"],
                        payload["height_cm"],
                        payload["activity_level"],
                        payload["goal"],
                        payload.get("meal_preference"),
                        payload.get("meal_source"),
                        payload.get("daily_budget"),
                    ),
                )
                for restriction in profile.dietary_restrictions:
                    connection.execute(
                        "INSERT INTO user_dietary_restrictions (user_id, restriction) VALUES (?, ?)",
                        (user_id, restriction),
                    )
                for ingredient in payload.get("pantry_ingredients", []):
                    connection.execute(
                        "INSERT INTO user_pantry_ingredients (user_id, ingredient) VALUES (?, ?)",
                        (user_id, str(ingredient)),
                    )
        except sqlite3.Error as error:
            raise StorageError(f"Could not save profile: {error}") from error
        finally:
            connection.close()

    def find_profiles(self, key: str, value: Any) -> Dict[str, UserProfile]:
        rows = self._execute(
            "SELECT user_id FROM users ORDER BY rowid",
            fetch=True,
        )
        matches = {}
        for (user_id,) in rows:
            data = self._load_user_record(user_id)
            if not data:
                continue
            if data.get(key) == value:
                matches[user_id] = UserProfile.from_dict(data)
        return matches

    def update_profile(self, username: str, changes: Dict[str, Any]) -> bool:
        user_id = self._user_id_for_username(username)
        if user_id is None:
            return False

        for key, value in changes.items():
            if key == "dietary_restrictions":
                self._replace_array_values("user_dietary_restrictions", user_id, "restriction", value)
                continue
            if key == "pantry_ingredients":
                self._replace_array_values("user_pantry_ingredients", user_id, "ingredient", value)
                continue
            if key in self._USER_COLUMNS:
                self._execute(
                    f"UPDATE users SET {key} = ? WHERE username = ?",
                    (value, username),
                )

        return True

    def save_plan(self, plan: MealPlan, user_id: str) -> None:
        plan_id = str(uuid.uuid4())
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT INTO meal_plans (plan_id, username, is_complete, daily_calorie_target, total_calories, "
                    "daily_budget, total_cost, within_budget, explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan_id,
                        user_id,
                        int(plan.is_complete),
                        plan.daily_calorie_target,
                        int(plan.total_calories),
                        plan.daily_budget,
                        plan.total_cost,
                        int(plan.within_budget),
                        plan.explanation,
                    ),
                )
                for slot in plan.missing_slots:
                    connection.execute(
                        "INSERT INTO meal_plan_missing_slots (plan_id, slot) VALUES (?, ?)",
                        (plan_id, slot),
                    )
                for slot, meal in plan.meals.items():
                    cursor = connection.execute(
                        "INSERT INTO meals (plan_id, slot, name, source_type, calorie_range, calorie_tag, "
                        "protein_tag, carbs_tag, fat_tag, cost_tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            plan_id,
                            slot,
                            meal.name,
                            meal.source_type,
                            meal.to_dict().get("calorie_range"),
                            meal.calorie_tag,
                            meal.to_dict().get("protein_tag"),
                            meal.to_dict().get("carbs_tag"),
                            meal.to_dict().get("fat_tag"),
                            meal.cost_tier,
                        ),
                    )
                    meal_id = cursor.lastrowid
                    for ingredient in meal.ingredient_names:
                        connection.execute(
                            "INSERT INTO meal_ingredients (meal_id, ingredient) VALUES (?, ?)",
                            (meal_id, ingredient),
                        )
                    if meal.portion_size or meal.recipe_steps:
                        connection.execute(
                            "INSERT INTO meal_recipes (meal_id, portion_size) VALUES (?, ?)",
                            (meal_id, meal.portion_size),
                        )
                    for step_no, instruction in enumerate(meal.recipe_steps, start=1):
                        connection.execute(
                            "INSERT INTO meal_recipe_steps (meal_id, step_no, instruction) VALUES (?, ?, ?)",
                            (meal_id, step_no, instruction),
                        )
        except sqlite3.Error as error:
            raise StorageError(f"Could not save plan: {error}") from error
        finally:
            connection.close()

    def plans_for_user(self, user_id: str) -> Dict[str, MealPlan]:
        rows = self._execute(
            "SELECT plan_id, username, is_complete, daily_calorie_target, total_calories, daily_budget, "
            "total_cost, within_budget, explanation FROM meal_plans WHERE username = ? ORDER BY rowid",
            (user_id,),
            fetch=True,
        )
        plans = {}
        for row in rows:
            plan_id = row["plan_id"]
            plan = MealPlan(row["daily_calorie_target"], row["daily_budget"])
            plan._total_calories = row["total_calories"] or 0
            plan._stored_total_cost = row["total_cost"]
            plan._missing_slots = [
                item[0] for item in self._execute(
                    "SELECT slot FROM meal_plan_missing_slots WHERE plan_id = ? ORDER BY slot",
                    (plan_id,),
                    fetch=True,
                )
            ]
            meal_rows = self._execute(
                "SELECT meal_id, slot, name, source_type, calorie_range, calorie_tag, protein_tag, carbs_tag, "
                "fat_tag, cost_tier FROM meals WHERE plan_id = ? ORDER BY meal_id",
                (plan_id,),
                fetch=True,
            )
            for meal_row in meal_rows:
                ingredients = [
                    item[0] for item in self._execute(
                        "SELECT ingredient FROM meal_ingredients WHERE meal_id = ? ORDER BY rowid",
                        (meal_row["meal_id"],),
                        fetch=True,
                    )
                ]
                portion_rows = self._execute(
                    "SELECT portion_size FROM meal_recipes WHERE meal_id = ?",
                    (meal_row["meal_id"],),
                    fetch=True,
                )
                recipe_steps = [
                    item[0] for item in self._execute(
                        "SELECT instruction FROM meal_recipe_steps WHERE meal_id = ? ORDER BY step_no",
                        (meal_row["meal_id"],),
                        fetch=True,
                    )
                ]
                plan._meals[meal_row["slot"]] = PlannedMeal(
                    slot=meal_row["slot"],
                    name=meal_row["name"],
                    source_type=meal_row["source_type"],
                    calorie_range=meal_row["calorie_range"],
                    calorie_tag=meal_row["calorie_tag"],
                    protein_tag=meal_row["protein_tag"],
                    carbs_tag=meal_row["carbs_tag"],
                    fat_tag=meal_row["fat_tag"],
                    cost_tier=meal_row["cost_tier"],
                    ingredients=ingredients,
                    calories=None,
                    cost=None,
                    recipe_steps=recipe_steps,
                    portion_size=portion_rows[0][0] if portion_rows else "",
                )
            plans[plan_id] = plan
        return plans