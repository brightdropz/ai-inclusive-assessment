
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal
import json
import os
import sqlite3
import uuid

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "assessment.db"

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "uci_student_performance"
    / "csv"
    / "student-por.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_student_risk_tuned.joblib"
)

VALIDATION_REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "random_forest_validation.json"
)

# Load the trained ML model once when the backend starts.
ML_MODEL_PACKAGE = None
ML_MODEL_ERROR = None

try:
    if MODEL_PATH.exists():
        ML_MODEL_PACKAGE = joblib.load(MODEL_PATH)
        print("Random Forest model loaded successfully.")
    else:
        ML_MODEL_ERROR = "Trained model file was not found."
        print(ML_MODEL_ERROR)

except Exception as error:
    ML_MODEL_ERROR = str(error)
    print(f"Random Forest loading error: {error}")


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Inclusive Assessment System API",
    description=(
        "An adaptive and accessible assessment platform "
        "with an experimental student-performance prediction module."
    ),
    version="2.0.0",
)

# Local development origins.
# Additional public frontend origins can be configured later
# through the FRONTEND_ORIGINS environment variable.

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

extra_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins + extra_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# 3. SAMPLE ADAPTIVE QUESTION BANK
# 15 demonstration questions:
# 5 Easy, 5 Medium and 5 Hard.
#
# This question bank is separate from the UCI ML dataset.
# ============================================================

QUESTIONS = [

    # ---------------- EASY QUESTIONS ----------------

    {
        "id": 1,
        "category": "Mathematics",
        "difficulty": "Easy",
        "prompt": "What is 15 + 27?",
        "options": ["32", "40", "42", "45"],
        "correct": 2,
        "explanation": "15 + 27 = 42.",
    },
    {
        "id": 2,
        "category": "General Knowledge",
        "difficulty": "Easy",
        "prompt": "Which planet is known as the Red Planet?",
        "options": ["Venus", "Mars", "Jupiter", "Mercury"],
        "correct": 1,
        "explanation": "Mars is known as the Red Planet.",
    },
    {
        "id": 3,
        "category": "Mathematics",
        "difficulty": "Easy",
        "prompt": "What is 9 multiplied by 6?",
        "options": ["45", "54", "56", "64"],
        "correct": 1,
        "explanation": "9 multiplied by 6 equals 54.",
    },
    {
        "id": 4,
        "category": "Science",
        "difficulty": "Easy",
        "prompt": "Which of these animals is a mammal?",
        "options": ["Shark", "Dolphin", "Trout", "Octopus"],
        "correct": 1,
        "explanation": (
            "Dolphins are mammals because they breathe air "
            "and feed milk to their young."
        ),
    },
    {
        "id": 5,
        "category": "Mathematics",
        "difficulty": "Easy",
        "prompt": "What is 100 divided by 4?",
        "options": ["20", "25", "30", "40"],
        "correct": 1,
        "explanation": "100 divided by 4 equals 25.",
    },

    # ---------------- MEDIUM QUESTIONS ----------------

    {
        "id": 6,
        "category": "Logical Reasoning",
        "difficulty": "Medium",
        "prompt": "What number comes next: 2, 4, 8, 16, ...?",
        "options": ["18", "24", "32", "36"],
        "correct": 2,
        "explanation": (
            "Each number is multiplied by 2. "
            "The next number is 32."
        ),
    },
    {
        "id": 7,
        "category": "Mathematics",
        "difficulty": "Medium",
        "prompt": "What is 25% of 80?",
        "options": ["15", "20", "25", "30"],
        "correct": 1,
        "explanation": "25% of 80 is 20.",
    },
    {
        "id": 8,
        "category": "Logical Reasoning",
        "difficulty": "Medium",
        "prompt": "Find the next number: 3, 6, 11, 18, 27, ...",
        "options": ["36", "37", "38", "40"],
        "correct": 2,
        "explanation": (
            "The differences are 3, 5, 7, 9 and then 11. "
            "The next number is 38."
        ),
    },
    {
        "id": 9,
        "category": "Mathematics",
        "difficulty": "Medium",
        "prompt": (
            "A car travels 150 kilometres in 3 hours. "
            "What is its average speed?"
        ),
        "options": [
            "40 km/h",
            "50 km/h",
            "60 km/h",
            "75 km/h",
        ],
        "correct": 1,
        "explanation": (
            "Average speed = distance divided by time "
            "= 150 / 3 = 50 km/h."
        ),
    },
    {
        "id": 10,
        "category": "Logical Reasoning",
        "difficulty": "Medium",
        "prompt": (
            "If BOOK is coded as CPPL, how is WORD coded "
            "using the same pattern?"
        ),
        "options": ["XPSE", "VNSC", "XQSE", "WQRF"],
        "correct": 0,
        "explanation": (
            "Each letter moves forward by one position. "
            "WORD becomes XPSE."
        ),
    },

    # ---------------- HARD QUESTIONS ----------------

    {
        "id": 11,
        "category": "Mathematics",
        "difficulty": "Hard",
        "prompt": (
            "A number is increased by 20% and becomes 120. "
            "What was the original number?"
        ),
        "options": ["90", "100", "110", "96"],
        "correct": 1,
        "explanation": (
            "120 divided by 1.2 gives the original number, "
            "which is 100."
        ),
    },
    {
        "id": 12,
        "category": "Logical Reasoning",
        "difficulty": "Hard",
        "prompt": (
            "If all roses are flowers and some flowers fade quickly, "
            "which statement must be true?"
        ),
        "options": [
            "All roses fade quickly",
            "Some roses fade quickly",
            "All roses are flowers",
            "No roses fade quickly",
        ],
        "correct": 2,
        "explanation": (
            "All roses being flowers is explicitly stated "
            "in the question."
        ),
    },
    {
        "id": 13,
        "category": "Mathematics",
        "difficulty": "Hard",
        "prompt": (
            "A fair coin is tossed twice. "
            "What is the probability of getting two heads?"
        ),
        "options": ["1/2", "1/3", "1/4", "3/4"],
        "correct": 2,
        "explanation": (
            "The possible outcomes are HH, HT, TH and TT. "
            "Only one gives two heads, so the probability is 1/4."
        ),
    },
    {
        "id": 14,
        "category": "Mathematics",
        "difficulty": "Hard",
        "prompt": "Solve the equation: 3x + 7 = 28.",
        "options": ["5", "6", "7", "8"],
        "correct": 2,
        "explanation": "3x = 21, therefore x = 7.",
    },
    {
        "id": 15,
        "category": "Mathematics",
        "difficulty": "Hard",
        "prompt": "Find the average of 4, 8, 10, 12 and 16.",
        "options": ["8", "9", "10", "11"],
        "correct": 2,
        "explanation": (
            "The sum is 50. Dividing by 5 gives an average of 10."
        ),
    },
]

QUESTIONS_BY_ID = {
    question["id"]: question
    for question in QUESTIONS
}


# ============================================================
# 4. DATABASE CONFIGURATION
# ============================================================

@contextmanager
def get_db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database():
    with get_db() as connection:

        connection.execute("""
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

        connection.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                user_id TEXT PRIMARY KEY,
                large_text INTEGER NOT NULL DEFAULT 0,
                high_contrast INTEGER NOT NULL DEFAULT 0,
                reduced_motion INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)


initialize_database()


# ============================================================
# 5. REQUEST MODELS
# ============================================================

class StartSessionRequest(BaseModel):
    user_id: str = Field(
        default="demo-user",
        min_length=1,
        max_length=100,
    )


class AnswerRequest(BaseModel):
    question_id: int = Field(ge=1)
    selected_option: int = Field(ge=0, le=3)


class PreferencesRequest(BaseModel):
    large_text: bool = False
    high_contrast: bool = False
    reduced_motion: bool = False


# Supplementary Random Forest input:
# These are exactly the 16 features used by ml_train.py
# and ml_validate.py.

class StudentRiskRequest(BaseModel):

    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)

    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]

    traveltime: int = Field(ge=1, le=4)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=1000)


# ============================================================
# 6. ADAPTIVE ASSESSMENT LOGIC
# ============================================================

ABILITY_LEVELS = {
    "Easy": 1,
    "Medium": 2,
    "Hard": 3,
}


def get_difficulty(ability: int) -> str:
    if ability <= 1:
        return "Easy"

    if ability >= 3:
        return "Hard"

    return "Medium"


def get_next_question(ability: int, used_ids: list):
    target_level = max(1, min(3, ability))

    # Prefer a question at the target difficulty.
    for question in QUESTIONS:
        if (
            question["difficulty"] == get_difficulty(target_level)
            and question["id"] not in used_ids
        ):
            return question

    # If the target difficulty is exhausted, choose the
    # nearest available difficulty instead of randomly
    # falling back to Easy.
    unused_questions = [
        question
        for question in QUESTIONS
        if question["id"] not in used_ids
    ]

    if not unused_questions:
        return None

    return min(
        unused_questions,
        key=lambda question: (
            abs(
                ABILITY_LEVELS[question["difficulty"]]
                - target_level
            ),
            -ABILITY_LEVELS[question["difficulty"]],
        ),
    )


def public_question(question):
    if question is None:
        return None

    # The correct answer is never sent to the frontend.
    return {
        "id": question["id"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "prompt": question["prompt"],
        "options": question["options"],
    }


def calculate_accuracy(score: int, answered_count: int) -> float:
    if answered_count == 0:
        return 0.0

    return round(
        (score / answered_count) * 100,
        2,
    )


# ============================================================
# 7. HEALTH AND HOME ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI Inclusive Assessment System API",
        "version": "2.0.0",
        "documentation": "/docs",
        "status": "running",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Inclusive Assessment System",
        "database": "SQLite",
        "random_forest": (
            "loaded"
            if ML_MODEL_PACKAGE is not None
            else "unavailable"
        ),
    }


# ============================================================
# 8. DATASET INFORMATION ENDPOINT
# Useful for showing the dataset details during evaluation.
# ============================================================

@app.get("/api/dataset/info")
def dataset_information():

    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset CSV file was not found.",
        )

    try:
        dataset = pd.read_csv(DATASET_PATH, sep=";")

        return {
            "dataset_name": (
                "UCI Student Performance - Portuguese subset"
            ),
            "source": (
                "https://archive.ics.uci.edu/dataset/320/"
                "student+performance"
            ),
            "records": int(len(dataset)),
            "raw_columns": int(len(dataset.columns)),
            "missing_values": int(
                dataset.isna().sum().sum()
            ),
            "selected_ml_features": 16,
            "target": "Pass if G3 >= 10; otherwise Fail",
            "grade_columns": ["G1", "G2", "G3"],
            "file": "student-por.csv",
            "license": "CC BY 4.0",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not inspect dataset: {str(error)}",
        )


# ============================================================
# 9. START ASSESSMENT SESSION
# ============================================================

@app.post("/api/sessions/start")
def start_session(request: StartSessionRequest):

    session_id = str(uuid.uuid4())

    # Start every learner at Medium difficulty.
    initial_ability = 2

    first_question = get_next_question(
        initial_ability,
        [],
    )

    if first_question is None:
        raise HTTPException(
            status_code=500,
            detail="No assessment questions are available.",
        )

    created_at = datetime.now(timezone.utc).isoformat()

    with get_db() as connection:

        connection.execute(
            """
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                request.user_id,
                initial_ability,
                0,
                0,
                json.dumps([first_question["id"]]),
                first_question["id"],
                0,
                created_at,
            ),
        )

    return {
        "session_id": session_id,
        "message": "Assessment session started.",
        "question_number": 1,
        "total_questions": 5,
        "ability_level": get_difficulty(initial_ability),
        "question": public_question(first_question),
    }


# ============================================================
# 10. SUBMIT ANSWER AND ADAPT DIFFICULTY
# ============================================================

@app.post("/api/sessions/{session_id}/answers")
def submit_answer(
    session_id: str,
    request: AnswerRequest,
):

    with get_db() as connection:

        session = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Assessment session not found.",
            )

        if session["is_finished"]:
            raise HTTPException(
                status_code=409,
                detail="This assessment is already finished.",
            )

        current_question_id = session["current_question_id"]

        if current_question_id != request.question_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The submitted question does not match "
                    "the current question."
                ),
            )

        question = QUESTIONS_BY_ID.get(current_question_id)

        if question is None:
            raise HTTPException(
                status_code=500,
                detail="Question data is unavailable.",
            )

        if request.selected_option >= len(question["options"]):
            raise HTTPException(
                status_code=422,
                detail="Invalid answer option.",
            )

        # Evaluate answer on the backend.
        is_correct = (
            request.selected_option == question["correct"]
        )

        previous_ability = session["ability"]

        if is_correct:
            new_ability = min(3, previous_ability + 1)
        else:
            new_ability = max(1, previous_ability - 1)

        new_score = (
            session["score"] + (1 if is_correct else 0)
        )

        new_answered_count = session["answered_count"] + 1

        used_ids = json.loads(session["used_ids"])

        next_question = None
        is_finished = False

        # The current demo assessment has five questions.
        if new_answered_count >= 5:
            is_finished = True

        else:
            next_question = get_next_question(
                new_ability,
                used_ids,
            )

            if next_question is None:
                is_finished = True

            else:
                used_ids.append(next_question["id"])

        next_question_id = (
            next_question["id"]
            if next_question is not None
            else None
        )

        connection.execute(
            """
            UPDATE sessions
            SET ability = ?,
                score = ?,
                answered_count = ?,
                used_ids = ?,
                current_question_id = ?,
                is_finished = ?
            WHERE session_id = ?
            """,
            (
                new_ability,
                new_score,
                new_answered_count,
                json.dumps(used_ids),
                next_question_id,
                int(is_finished),
                session_id,
            ),
        )

    return {
        "correct": is_correct,
        "explanation": question["explanation"],
        "score": new_score,
        "answered_count": new_answered_count,
        "accuracy": calculate_accuracy(
            new_score,
            new_answered_count,
        ),
        "ability_level": get_difficulty(new_ability),
        "finished": is_finished,
        "next_question": public_question(next_question),
    }


# ============================================================
# 11. GET ASSESSMENT RESULTS
# ============================================================

@app.get("/api/sessions/{session_id}/results")
def get_results(session_id: str):

    with get_db() as connection:

        session = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Assessment session not found.",
        )

    score = session["score"]
    answered_count = session["answered_count"]

    return {
        "session_id": session_id,
        "user_id": session["user_id"],
        "score": score,
        "answered_count": answered_count,
        "accuracy": calculate_accuracy(
            score,
            answered_count,
        ),
        "ability_level": get_difficulty(
            session["ability"]
        ),
        "finished": bool(session["is_finished"]),
        "created_at": session["created_at"],
    }


# ============================================================
# 12. ACCESSIBILITY PREFERENCES
# ============================================================

@app.get("/api/users/{user_id}/preferences")
def get_preferences(user_id: str):

    with get_db() as connection:

        row = connection.execute(
            """
            SELECT *
            FROM preferences
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return {
            "user_id": user_id,
            "large_text": False,
            "high_contrast": False,
            "reduced_motion": False,
        }

    return {
        "user_id": user_id,
        "large_text": bool(row["large_text"]),
        "high_contrast": bool(row["high_contrast"]),
        "reduced_motion": bool(row["reduced_motion"]),
    }


@app.post("/api/users/{user_id}/preferences")
def save_preferences(
    user_id: str,
    request: PreferencesRequest,
):

    updated_at = datetime.now(timezone.utc).isoformat()

    with get_db() as connection:

        connection.execute(
            """
            INSERT INTO preferences (
                user_id,
                large_text,
                high_contrast,
                reduced_motion,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(user_id) DO UPDATE SET
                large_text = excluded.large_text,
                high_contrast = excluded.high_contrast,
                reduced_motion = excluded.reduced_motion,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                int(request.large_text),
                int(request.high_contrast),
                int(request.reduced_motion),
                updated_at,
            ),
        )

    return {
        "message": "Accessibility preferences saved.",
        "user_id": user_id,
        "large_text": request.large_text,
        "high_contrast": request.high_contrast,
        "reduced_motion": request.reduced_motion,
    }


# ============================================================
# 13. RANDOM FOREST MODEL INFORMATION
# ============================================================

@app.get("/api/model/info")
def model_information():

    if ML_MODEL_PACKAGE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Random Forest model is unavailable. "
                "Run ml_validate.py first."
            ),
        )

    return {
        "model": "RandomForestClassifier",
        "dataset": (
            "UCI Student Performance - Portuguese subset"
        ),
        "input_features": len(
            ML_MODEL_PACKAGE["feature_columns"]
        ),
        "feature_columns": (
            ML_MODEL_PACKAGE["feature_columns"]
        ),
        "target": ML_MODEL_PACKAGE["target_definition"],
        "fail_probability_threshold": (
            ML_MODEL_PACKAGE["fail_probability_threshold"]
        ),
        "status": "loaded",
        "purpose": "Experimental supplementary prediction",
    }


# ============================================================
# 14. RANDOM FOREST EVALUATION METRICS
# ============================================================

@app.get("/api/model/metrics")
def model_metrics():

    if not VALIDATION_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Model validation report was not found.",
        )

    try:
        with open(
            VALIDATION_REPORT_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

        return report

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read model metrics: {str(error)}",
        )


# ============================================================
# 15. RANDOM FOREST STUDENT-PERFORMANCE PREDICTION
# Experimental supplementary AI feature.
# This must NOT be used for high-stakes student decisions.
# ============================================================

@app.post("/api/predictions/student-risk")
def predict_student_risk(request: StudentRiskRequest):

    if ML_MODEL_PACKAGE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Random Forest model is unavailable. "
                "Train and validate the model first."
            ),
        )

    try:

        feature_columns = (
            ML_MODEL_PACKAGE["feature_columns"]
        )

        model = ML_MODEL_PACKAGE["pipeline"]

        # Convert validated request into the same feature
        # structure used during model training.
        input_data = pd.DataFrame(
            [
                {
                    column: getattr(request, column)
                    for column in feature_columns
                }
            ],
            columns=feature_columns,
        )

        probabilities = model.predict_proba(
            input_data
        )[0]

        # Class 0 = Fail; Class 1 = Pass.
        fail_probability = float(probabilities[0])
        pass_probability = float(probabilities[1])

        threshold = float(
            ML_MODEL_PACKAGE[
                "fail_probability_threshold"
            ]
        )

        needs_additional_review = (
            fail_probability >= threshold
        )

        return {
            "model": "Random Forest",
            "prediction": (
                "Needs additional review"
                if needs_additional_review
                else "Lower predicted risk"
            ),
            "predicted_class": (
                "Fail"
                if needs_additional_review
                else "Pass"
            ),
            "fail_probability": round(
                fail_probability * 100,
                2,
            ),
            "pass_probability": round(
                pass_probability * 100,
                2,
            ),
            "risk_threshold": threshold,
            "needs_additional_review": (
                needs_additional_review
            ),
            "notice": (
                "Experimental educational prediction for "
                "demonstration only. This output is not a "
                "diagnosis or a decision about a student's "
                "ability and must not replace teacher judgement."
            ),
        }

    except Exception as error:

        print(f"Student prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail="The student prediction could not be completed.",
        )


# ============================================================
# 16. APPLICATION STARTUP INFORMATION
# ============================================================

print("==============================================")
print("AI INCLUSIVE ASSESSMENT SYSTEM")
print("Backend application initialized.")
print(f"Question bank: {len(QUESTIONS)} questions")

if ML_MODEL_PACKAGE is not None:
    print("Random Forest: loaded")
else:
    print("Random Forest: unavailable")

print(f"Database: {DB_PATH}")
print("==============================================")
