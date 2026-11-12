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

DEFAULT_STORAGE_BACKEND = "sqlite"


def get_storage_backend() -> str:
    """Resolve the storage backend from the current environment.

    This is evaluated lazily so .env files loaded at app startup are honoured,
    instead of freezing the backend at import time before dotenv runs.
    """
    return os.getenv("CALORA_STORAGE", DEFAULT_STORAGE_BACKEND).strip().lower()


# "json" (default) or "sqlite"; chosen with the CALORA_STORAGE environment variable
STORAGE_BACKEND = get_storage_backend()

# Background plan generation (threading)
PLAN_WORKER_THREADS = 4
MAX_TRACKED_JOBS = 200


# Parallel AI models (threading): two Gemini models are asked at the same time and
# the first valid answer wins. Evaluated lazily so values from .env are honoured.
# Check the fallback model name is available to your API key in Google AI Studio.
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"
PARALLEL_TIMEOUT_SECONDS = 60.0


def get_fallback_model() -> str:
    """Second model for the parallel run (env GEMINI_FALLBACK_MODEL)."""
    return os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL).strip()


def is_parallel_ai_enabled() -> bool:
    """On by default; set CALORA_PARALLEL_AI=0 to use the single primary model."""
    return os.getenv("CALORA_PARALLEL_AI", "1").strip().lower() not in ("0", "false", "no", "off")
