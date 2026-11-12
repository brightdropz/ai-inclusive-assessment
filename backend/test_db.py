
from pathlib import Path
import os

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise SystemExit("DATABASE_URL is missing from backend/.env")

try:
    with psycopg.connect(database_url, connect_timeout=15) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user;"
            )
            database, user = cursor.fetchone()

    print("NEON POSTGRESQL CONNECTION SUCCESS")
    print("Database:", database)
    print("Role:", user)
    print("Connection verified successfully.")

except Exception as error:
    print("NEON CONNECTION FAILED")
    print("Error:", error)
    raise SystemExit(1)
