
from pathlib import Path
import os
import sqlite3

import psycopg
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

SQLITE_PATH = BASE_DIR / "assessment.db"

if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is missing from backend/.env")

if not SQLITE_PATH.exists():
    raise SystemExit("Local assessment.db was not found.")


# --------------------------------------------------
# READ EXISTING SQLITE DATA
# --------------------------------------------------

sqlite_conn = sqlite3.connect(SQLITE_PATH)
sqlite_conn.row_factory = sqlite3.Row

def read_table(table_name, columns):
    exists = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()

    if not exists:
        return []

    query = (
        f"SELECT {', '.join(columns)} FROM {table_name}"
    )

    return [
        tuple(row[column] for column in columns)
        for row in sqlite_conn.execute(query).fetchall()
    ]


session_columns = [
    "session_id",
    "user_id",
    "ability",
    "score",
    "answered_count",
    "used_ids",
    "current_question_id",
    "is_finished",
    "created_at",
]

preference_columns = [
    "user_id",
    "large_text",
    "high_contrast",
    "reduced_motion",
    "updated_at",
]

sessions = read_table("sessions", session_columns)
preferences = read_table("preferences", preference_columns)

sqlite_conn.close()

print("LOCAL DATABASE RECORDS")
print("Assessment sessions:", len(sessions))
print("Accessibility preferences:", len(preferences))


# --------------------------------------------------
# CREATE POSTGRESQL TABLES AND MIGRATE RECORDS
# --------------------------------------------------

with psycopg.connect(DATABASE_URL, connect_timeout=20) as conn:

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            ability INTEGER NOT NULL DEFAULT 2,
            score INTEGER NOT NULL DEFAULT 0,
            answered_count INTEGER NOT NULL DEFAULT 0,
            used_ids TEXT NOT NULL DEFAULT '[]',
            current_question_id INTEGER,
            is_finished INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT PRIMARY KEY,
            large_text INTEGER NOT NULL DEFAULT 0,
            high_contrast INTEGER NOT NULL DEFAULT 0,
            reduced_motion INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)

    session_sql = """
        INSERT INTO sessions (
            session_id,
            user_id,
            ability,
            score,
            answered_count,
            used_ids,
            current_question_id,
            is_finished,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id) DO NOTHING
    """

    preference_sql = """
        INSERT INTO preferences (
            user_id,
            large_text,
            high_contrast,
            reduced_motion,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO NOTHING
    """

    migrated_sessions = 0
    migrated_preferences = 0

    for row in sessions:
        cursor = conn.execute(session_sql, row)
        migrated_sessions += max(cursor.rowcount, 0)

    for row in preferences:
        cursor = conn.execute(preference_sql, row)
        migrated_preferences += max(cursor.rowcount, 0)

    # Verify records in PostgreSQL.
    target_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions"
    ).fetchone()[0]

    target_preferences = conn.execute(
        "SELECT COUNT(*) FROM preferences"
    ).fetchone()[0]


# --------------------------------------------------
# REPORT MIGRATION STATUS
# --------------------------------------------------

print("\nPOSTGRESQL MIGRATION COMPLETED")
print("New assessment sessions copied:", migrated_sessions)
print("New preferences copied:", migrated_preferences)
print("Total sessions now in Neon:", target_sessions)
print("Total preferences now in Neon:", target_preferences)
print("\nLocal SQLite backup remains unchanged.")
