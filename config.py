"""Central settings. Secrets (the API key) are read from the environment, never stored here."""
import os

GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
RECOMMENDATION_COUNT = 10

DATA_FILE_ROOT = "data/"
USER_FILE = os.path.join(DATA_FILE_ROOT, "user_profiles.json")
MEAL_PLAN_FILE = os.path.join(DATA_FILE_ROOT, "meal_plans.json")
SQLITE_DB_FILE = os.path.join(DATA_FILE_ROOT, "calora.db")

# "json" (default) or "sqlite"; chosen with the CALORA_STORAGE environment variable
STORAGE_BACKEND = os.getenv("CALORA_STORAGE", "json").strip().lower()

# Background plan generation (threading)
PLAN_WORKER_THREADS = 4
MAX_TRACKED_JOBS = 200
