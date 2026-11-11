"""Central settings. Secrets (the API key) are read from the environment, never stored here."""
import os

GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
RECOMMENDATION_COUNT = 10

USER_FILE = os.path.join("data", "user_profiles.json")
MEAL_PLAN_FILE = os.path.join("data", "meal_plans.json")
SQLITE_DB_FILE = os.path.join("data", "calora.db")


def get_storage_backend() -> str:
    """Resolve the storage backend after the app loads its environment."""
    return os.getenv("CALORA_STORAGE", "sqlite").strip().lower()

# Background plan generation (threading)
PLAN_WORKER_THREADS = 4
MAX_TRACKED_JOBS = 200