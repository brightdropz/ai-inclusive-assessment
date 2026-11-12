
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

DATA_PATH = (
    BASE / "data" / "uci_student_performance"
    / "csv" / "student-por.csv"
)

MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher",
    "internet", "traveltime", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

RANDOM_STATE = 42
N_ESTIMATORS = 300
TEST_SIZE = 0.20
PASS_THRESHOLD_GRADE = 10


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";")

X = df[FEATURES].copy()

# 1 = Pass; 0 = Fail
y = (df["G3"] >= PASS_THRESHOLD_GRADE).astype(int)

# Separate training data and untouched test data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== VALIDATION DATA ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training pass records:", int((y_train == 1).sum()))
print("Training fail records:", int((y_train == 0).sum()))


# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# 4. RANDOM FOREST PIPELINE
# --------------------------------------------------

def create_model():
    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


# --------------------------------------------------
# 5. FIVE-FOLD CROSS-VALIDATION
# Generate out-of-fold predictions for training data.
# The test set is not used for threshold selection.
# --------------------------------------------------

print("\n========== FIVE-FOLD CROSS-VALIDATION ==========")

cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_model = create_model()

oof_probabilities = cross_val_predict(
    cv_model,
    X_train,
    y_train,
    cv=cross_validation,
    method="predict_proba",
    n_jobs=1,
)

# Probability of failing = 1 - probability of passing.
oof_fail_probability = oof_probabilities[:, 0]

y_train_fail = 1 - y_train.to_numpy()


# --------------------------------------------------
# 6. SELECT THRESHOLD USING TRAINING DATA ONLY
# Maximise F1-score for the Fail category.
# --------------------------------------------------

print("\n========== THRESHOLD SELECTION ==========")

thresholds = np.linspace(0.05, 0.95, 181)

threshold_results = []

for threshold in thresholds:
    predicted_fail = (
        oof_fail_probability >= threshold
    ).astype(int)

    fail_f1 = f1_score(
        y_train_fail,
        predicted_fail,
        zero_division=0,
    )

    fail_recall = recall_score(
        y_train_fail,
        predicted_fail,
        zero_division=0,
    )

    threshold_results.append({
        "threshold": float(threshold),
        "fail_f1": float(fail_f1),
        "fail_recall": float(fail_recall),
    })

# Select the threshold with the best cross-validated Fail F1.
# If several thresholds tie, favour higher Fail recall.
best_result = max(
    threshold_results,
    key=lambda item: (
        item["fail_f1"],
        item["fail_recall"],
    ),
)

selected_threshold = best_result["threshold"]

print("Selected risk threshold:", round(selected_threshold, 3))
print("Training OOF Fail F1:", round(best_result["fail_f1"], 4))
print("Training OOF Fail recall:", round(best_result["fail_recall"], 4))


# --------------------------------------------------
# 7. TRAIN FINAL MODEL ON ALL TRAINING RECORDS
# --------------------------------------------------

print("\n========== FINAL MODEL TRAINING ==========")

final_model = create_model()

final_model.fit(X_train, y_train)

print("Random Forest trained on the training partition.")


# --------------------------------------------------
# 8. EVALUATE ON UNTOUCHED TEST DATA
# --------------------------------------------------

print("\n========== HELD-OUT TEST EVALUATION ==========")

probabilities_pass = final_model.predict_proba(X_test)[:, 1]
probabilities_fail = 1 - probabilities_pass

# Default threshold for comparison.
default_threshold = 0.50

# Selected threshold from cross-validation.
y_pred_default = (
    probabilities_pass >= default_threshold
).astype(int)

y_pred_tuned = (
    probabilities_fail < selected_threshold
).astype(int)

y_test_fail = 1 - y_test.to_numpy()
y_pred_fail_tuned = 1 - y_pred_tuned


def evaluate_model(y_true, y_pred, probability_pass):
    y_true_fail = 1 - np.asarray(y_true)
    y_pred_fail = 1 - np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "pass_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "pass_recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "pass_f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "fail_precision": float(
            precision_score(
                y_true_fail,
                y_pred_fail,
                zero_division=0,
            )
        ),
        "fail_recall": float(
            recall_score(
                y_true_fail,
                y_pred_fail,
                zero_division=0,
            )
        ),
        "fail_f1": float(
            f1_score(
                y_true_fail,
                y_pred_fail,
                zero_division=0,
            )
        ),
        "roc_auc_pass": float(
            roc_auc_score(y_true, probability_pass)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


default_metrics = evaluate_model(
    y_test,
    y_pred_default,
    probabilities_pass,
)

tuned_metrics = evaluate_model(
    y_test,
    y_pred_tuned,
    probabilities_pass,
)

print("\nDEFAULT THRESHOLD (0.50)")
for key, value in default_metrics.items():
    print(f"{key}: {value}")

print("\nCROSS-VALIDATION TUNED THRESHOLD")
for key, value in tuned_metrics.items():
    print(f"{key}: {value}")

print("\nTUNED CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred_tuned,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 9. SAVE VALIDATION REPORT
# --------------------------------------------------

validation_report = {
    "dataset": "UCI Student Performance - Portuguese",
    "records": len(df),
    "features": FEATURES,
    "target": "Pass if G3 >= 10; otherwise Fail",
    "train_records": len(X_train),
    "test_records": len(X_test),
    "cross_validation_folds": 5,
    "random_state": RANDOM_STATE,
    "model": "RandomForestClassifier",
    "trees": N_ESTIMATORS,
    "threshold_selection": "Maximise Fail F1 on training out-of-fold predictions",
    "default_threshold": default_threshold,
    "selected_fail_probability_threshold": selected_threshold,
    "default_test_metrics": default_metrics,
    "tuned_test_metrics": tuned_metrics,
}

REPORT_PATH = REPORT_DIR / "random_forest_validation.json"

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(validation_report, file, indent=4)


# --------------------------------------------------
# 10. SAVE TUNED MODEL AND CONFIGURATION
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "random_forest_student_risk_tuned.joblib"

joblib.dump(
    {
        "pipeline": final_model,
        "feature_columns": FEATURES,
        "fail_probability_threshold": selected_threshold,
        "target_definition": "Pass if G3 >= 10; otherwise Fail",
        "dataset": "UCI Student Performance - Portuguese",
    },
    MODEL_PATH,
)

print("\n========== SAVED FILES ==========")
print("Validation report:", REPORT_PATH)
print("Tuned model:", MODEL_PATH)
print("\nValidation experiment completed.")


# --------------------------------------------------
# 11. IMPORTANT INTERPRETATION
# --------------------------------------------------

if tuned_metrics["fail_recall"] > default_metrics["fail_recall"]:
    print(
        "\nFail recall improved on the held-out test set."
    )
else:
    print(
        "\nFail recall did not improve on the held-out test set."
    )

print(
    "Threshold selection used training data only. "
    "The final metrics are from the held-out test partition."
)
