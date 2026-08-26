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
# DATABASE URL
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing from .env"
    )


# ============================================================
# CONNECTION
# ============================================================

def get_connection():

    return psycopg.connect(
        DATABASE_URL
    )