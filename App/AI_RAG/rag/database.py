import os
from dotenv import load_dotenv
from pathlib import Path
import psycopg


# ============================================================
# LOAD .ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

load_dotenv(BASE_DIR / ".env")


# ============================================================
# SUPABASE_DATABASE_URL
# ============================================================

SUPABASE_DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")

if not SUPABASE_DATABASE_URL:
    raise RuntimeError(
        "SUPABASE_DATABASE_URL is missing from .env"
    )


# ============================================================
# CONNECTION
# ============================================================

def get_connection():

    return psycopg.connect(
        SUPABASE_DATABASE_URL
    )