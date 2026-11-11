"""Central settings. Secrets (the API key) are read from the environment, never stored here."""
import os

GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
RECOMMENDATION_COUNT = 10

SQLITE_DB_FILE = os.path.join("data", "calora.db")

# Background plan generation (threading)
PLAN_WORKER_THREADS = 4
MAX_TRACKED_JOBS = 200