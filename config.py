"""Central settings. Secrets (the API key) are read from the environment, never stored here."""
import os
import math

GEMINI_MODEL = "gemini-3.1-flash-lite"
CLAUDE_MODEL = "claude-sonnet-5-5"
AI_REQUEST_TIMEOUT_SECONDS = 45.0
AI_TOTAL_TIMEOUT_SECONDS = 60.0
AI_MAX_OUTPUT_TOKENS = 8192


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip()


def get_claude_model() -> str:
    return os.getenv("CLAUDE_MODEL", CLAUDE_MODEL).strip()


def get_ai_mode() -> str:
    """Prefer multi when both keys exist; otherwise preserve the Gemini workflow."""
    mode = os.getenv("AI_MODE", "").strip().lower()
    if not mode:
        return "multi" if os.getenv("GEMINI_API_KEY") and os.getenv("ANTHROPIC_API_KEY") else "gemini"
    if mode not in ("gemini", "claude", "multi"):
        from manager.exceptions import InvalidInputError
        raise InvalidInputError("AI_MODE must be gemini, claude or multi.")
    return mode


def get_ai_timeout() -> float:
    return _positive_setting("AI_REQUEST_TIMEOUT_SECONDS", AI_REQUEST_TIMEOUT_SECONDS)


def get_ai_total_timeout() -> float:
    return _positive_setting("AI_TOTAL_TIMEOUT_SECONDS", AI_TOTAL_TIMEOUT_SECONDS)


def get_max_output_tokens() -> int:
    value = _positive_setting("AI_MAX_OUTPUT_TOKENS", AI_MAX_OUTPUT_TOKENS)
    if not value.is_integer():
        from manager.exceptions import InvalidInputError
        raise InvalidInputError("AI_MAX_OUTPUT_TOKENS must be a positive integer.")
    return int(value)


def _positive_setting(name: str, default: float) -> float:
    from manager.exceptions import InvalidInputError
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        raise InvalidInputError(f"{name} must be a positive finite number.") from None
    if not math.isfinite(value) or value <= 0:
        raise InvalidInputError(f"{name} must be a positive finite number.")
    return value


# Components are bounded to 0..1; the resulting score is out of 100.
RESPONSE_SCORE_WEIGHTS = {
    "calories": 30, "macros": 25, "budget": 15, "preference": 10,
    "source": 5, "pantry": 5, "variety": 5, "practicality": 5,
}
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
RECOMMENDATION_COUNT = 10

USER_FILE = os.path.join("data", "user_profiles.json")
MEAL_PLAN_FILE = os.path.join("data", "meal_plans.json")
SQLITE_DB_FILE = os.path.join("data", "calora.db")
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


def get_storage_backend() -> str:
    """Resolve the storage backend after the app loads its environment."""
    return os.getenv("CALORA_STORAGE", "sqlite").strip().lower()

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
