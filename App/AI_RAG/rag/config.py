"""NourFi - central configuration (RAG pipeline + chat backend).

Every tunable value of the project lives here, in one place.

* Values come from environment variables (.env), with a safe default in the code.
* Secrets (API keys, database URLs) are only READ from .env, never written in this file.
* Prompts and canned replies are in prompts.py (text, not settings).

To change a behaviour, change the value in .env (no code edit needed), e.g. TOP_K=8.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def resolve_database_url() -> tuple[str | None, str | None]:
    """Return (url, name_of_the_variable) of the RAG database, or (None, None)."""
    for name in DATABASE_URL_VARIABLES:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())


# ---------------------------------------------------------------------------
# 1. DATABASE (Supabase / pgvector, table public.rag_chunks)
# ---------------------------------------------------------------------------
DATABASE_URL_VARIABLES = ("SUPABASE_DATABASE_URL", "SUPABASE_DB_URL", "DATABASE_URL")  # first one set wins
DB_POOL_MAX = _int("DB_POOL_MAX", 5)          # max pooled connections
DB_POOL_MAX_IDLE = 60                          # seconds; Supabase closes idle connections, recycle before
DB_POOL_MAX_LIFETIME = 900                     # seconds
DB_CONNECT_TIMEOUT = 30                        # seconds


# ---------------------------------------------------------------------------
# 2. EMBEDDING + RETRIEVAL
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = _str("MODEL_NAME_3", "intfloat/multilingual-e5-small")
MAX_QUERY_LENGTH = _int("MAX_QUERY_LENGTH", 128)   # tokens kept when embedding the question
TOP_K = _int("TOP_K", 5)                           # excerpts sent to the LLM
CANDIDATE_MULTIPLIER = 3                           # candidates fetched = max(TOP_K * 3, CANDIDATE_MIN)
CANDIDATE_MIN = 15
MAX_CHUNK_CHARS = _int("MAX_CHUNK_CHARS", 900)     # characters of each excerpt sent to the LLM
# Similarity gate. 0 = disabled. Calibrate from the logged "best_sim" values (calibrate.py).
MIN_SIM = _float("MIN_SIM", 0.0)
# "Give me all the methods" questions: best chunk of each of these AAOIFI standards.
# 8 murabaha, 9 ijarah, 10 salam, 11 istisna, 12 musharaka, 13 mudaraba, 17 sukuk.
CORE_STANDARDS = _csv("CORE_STANDARDS", "8,9,10,11,12,13,17")
BROAD_MIN_STANDARDS = 3                            # broad mode needs at least this many standards in the DB
TRANSLATE_ARABIC = _bool("TRANSLATE_ARABIC", True)  # Arabic question -> English before embedding
TYPO_CUTOFF = 0.88                                 # difflib similarity needed to auto-correct a word
TYPO_MIN_LENGTH = 6                                # shorter words are never auto-corrected


# ---------------------------------------------------------------------------
# 3. LLM PROVIDERS
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")           # secret, from .env
GROQ_BASE_URL = _str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = _str("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = _str("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")  # separate model = separate quota
REASONING_EFFORT = _str("REASONING_EFFORT", "low")                  # only sent to gpt-oss models
SUMMARY_MODEL = _str("SUMMARY_MODEL", "")                           # default: the fallback model

# Optional second provider (any OpenAI-compatible API). All three must be set to enable it.
SECOND_API_KEY = os.getenv("SECOND_API_KEY")
SECOND_BASE_URL = os.getenv("SECOND_BASE_URL")
SECOND_MODEL = os.getenv("SECOND_MODEL")

LLM_REQUEST_TIMEOUT = 45.0                         # seconds per HTTP call


# ---------------------------------------------------------------------------
# 4. GENERATION + RATE LIMITS
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE = 0.1
MAX_GENERATION_TOKENS = min(_int("MAX_GENERATION_TOKENS", 700), 2500)
RETRY_MAX_GENERATION_TOKENS = 4000                 # cap for the single "empty answer" retry
MAX_RETRIES = _int("MAX_RETRIES", 4)
LLM_MAX_CONCURRENCY = _int("LLM_MAX_CONCURRENCY", 2)   # simultaneous LLM calls
MAX_WAIT_SECONDS = _float("MAX_WAIT_SECONDS", 20)      # never wait longer than this on a 429
SHORT_WAIT_SECONDS = _float("SHORT_WAIT_SECONDS", 8)   # a 429 shorter than this: just wait and retry
# Client-side tokens-per-minute limiter (Groq free tier: 8000 TPM counting prompt + max_tokens).
# 0 disables it (e.g. paid tier).
LLM_TPM_LIMIT = _int("LLM_TPM_LIMIT", 7000)
TPM_MAX_QUEUE_SECONDS = _float("TPM_MAX_QUEUE_SECONDS", 30)
TPM_WINDOW_SECONDS = 60
SECOND_COOLDOWN_SECONDS = _float("SECOND_COOLDOWN_SECONDS", 300)  # a failing 2nd provider is parked this long

# Small helper calls
TRANSLATE_MAX_TOKENS = 300
MEMORY_ANSWER_MAX_TOKENS = 350
SUMMARY_CALL_MAX_TOKENS = 200
SUMMARY_TEMPERATURE = 0.2


# ---------------------------------------------------------------------------
# 5. PROMPT BUDGET (tokens are what hit the TPM / TPD limits)
# ---------------------------------------------------------------------------
MODEL_CONTEXT_TOKENS = _int("MODEL_CONTEXT_TOKENS", 12000)
PROMPT_TOKEN_MARGIN = _int("PROMPT_TOKEN_MARGIN", 220)
PROMPT_TOKEN_BUDGET = _int("PROMPT_TOKEN_BUDGET", 4200)   # hard cap on prompt size
CHARS_PER_TOKEN = _int("CHARS_PER_TOKEN", 4)              # cheap token estimate
KEEP_RECENT_TURNS = _int("KEEP_RECENT_TURNS", 6)          # recent messages replayed to the model
HISTORY_ASSISTANT_MAX_CHARS = _int("HISTORY_ASSISTANT_MAX_CHARS", 350)  # old answers replayed short
SUMMARY_MAX_TOKENS = _int("SUMMARY_MAX_TOKENS", 180)                    # session summary cap
CROSS_CONTEXT_MAX_TOKENS = _int("CROSS_CONTEXT_MAX_TOKENS", 200)        # other-conversations cap


# ---------------------------------------------------------------------------
# 6. ANSWER CACHE (standalone questions only)
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", 3600)
CACHE_MAX_ENTRIES = _int("CACHE_MAX_ENTRIES", 500)


# ---------------------------------------------------------------------------
# 7. BACKEND (FastAPI, App/main.py)
# ---------------------------------------------------------------------------
CORS_ORIGINS = _csv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
EMAIL_CODE_EXPIRE_MINUTES = _int("EMAIL_CODE_EXPIRE_MINUTES", 15)     # verification code lifetime
RESET_TOKEN_EXPIRE_MINUTES = _int("RESET_TOKEN_EXPIRE_MINUTES", 30)   # password-reset link lifetime
SESSION_EXPIRED_DETAIL = "session_expired"   # 401 detail the frontend recognises (redirect to /login)

# Memory inside a conversation
SUMMARY_FREQUENCY = _int("SUMMARY_FREQUENCY", 4)          # summary refreshed every N messages
SUMMARY_WINDOW = _int("SUMMARY_WINDOW", 8)                # messages re-read to update the summary
RECENT_MESSAGES_LIMIT = _int("RECENT_MESSAGES_LIMIT", 6)  # latest raw messages sent to the pipeline
MAX_MESSAGE_CHARS = _int("MAX_MESSAGE_CHARS", 600)        # truncation per history message
MIN_SECONDS_BETWEEN_LLM_CALLS = _float("MIN_SECONDS_BETWEEN_GROQ_CALLS", 0.5)

# Memory across conversations
CROSS_CONV_MAX_CONVERSATIONS = _int("CROSS_CONV_MAX_CONVERSATIONS", 3)  # past conversations used
CROSS_CONV_DIGEST_QUESTIONS = _int("CROSS_CONV_DIGEST_QUESTIONS", 3)    # questions kept if no summary
CROSS_CONV_QUESTION_CHARS = _int("CROSS_CONV_QUESTION_CHARS", 160)      # truncation per question
CROSS_CONV_MAX_CHARS = _int("CROSS_CONV_MAX_CHARS", 800)                # total cap
CROSS_CONV_MIN_PRIOR_MESSAGES = 2   # a conversation shorter than this always gets cross-conversation context