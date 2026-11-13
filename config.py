"""Central settings. Secrets (the API key) are read from the environment, never stored here."""
import logging
import math
import os

from manager.exceptions import InvalidInputError

GEMINI_MODEL = "gemini-3.1-flash-lite"
CLAUDE_MODEL = "claude-sonnet-5-5"
AI_REQUEST_TIMEOUT_SECONDS = 60.0
CLAUDE_REQUEST_TIMEOUT_SECONDS = 120.0
AI_TOTAL_TIMEOUT_SECONDS = 450.0
AI_MAX_OUTPUT_TOKENS = 8192
CLAUDE_MAX_OUTPUT_TOKENS = 16384


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
        raise InvalidInputError("AI_MODE must be gemini, claude or multi.")
    return mode


def get_ai_timeout() -> float:
    return _positive_setting("AI_REQUEST_TIMEOUT_SECONDS", AI_REQUEST_TIMEOUT_SECONDS)


def get_ai_total_timeout() -> float:
    return _positive_setting("AI_TOTAL_TIMEOUT_SECONDS", AI_TOTAL_TIMEOUT_SECONDS)


def get_claude_timeout() -> float:
    """Allow more time for Claude to generate the complete ten-meal JSON response."""
    return _positive_setting("CLAUDE_REQUEST_TIMEOUT_SECONDS", CLAUDE_REQUEST_TIMEOUT_SECONDS)


def is_claude_backup_enabled() -> bool:
    """Replace failed Claude requests in multi mode unless explicitly disabled."""
    return os.getenv("CLAUDE_GEMINI_BACKUP", "1").strip().lower() not in ("0", "false", "no", "off")


def get_gemini_backup_model() -> str:
    """Use the configured primary Gemini model unless a separate backup is set."""
    return os.getenv("GEMINI_BACKUP_MODEL", "").strip() or get_gemini_model()


def get_max_output_tokens() -> int:
    return _positive_integer_setting("AI_MAX_OUTPUT_TOKENS", AI_MAX_OUTPUT_TOKENS)


def get_claude_max_output_tokens() -> int:
    """Leave enough output capacity for Claude's complete ten-meal response."""
    return _positive_integer_setting("CLAUDE_MAX_OUTPUT_TOKENS", CLAUDE_MAX_OUTPUT_TOKENS)


def _positive_integer_setting(name: str, default: int) -> int:
    value = _positive_setting(name, default)
    if not value.is_integer():
        raise InvalidInputError(f"{name} must be a positive integer.")
    return int(value)


def _positive_setting(name: str, default: float) -> float:
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


# Compatibility alias; AppFactory uses get_storage_backend() after dotenv loads.
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


def configure_logging() -> None:
    """Share readable application logs while suppressing SDK transport details."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    for sdk_logger in ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic", "google.genai", "google_genai"):
        logging.getLogger(sdk_logger).setLevel(logging.WARNING)
