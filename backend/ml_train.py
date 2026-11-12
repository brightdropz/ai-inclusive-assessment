
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "uci_student_performance"
    / "csv"
    / "student-por.csv"
)

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "random_forest_student_pass.joblib"
METRICS_PATH = REPORT_DIR / "random_forest_metrics.json"
MATRIX_PATH = REPORT_DIR / "confusion_matrix.csv"


# --------------------------------------------------
# 2. SELECTED INPUT FEATURES
# No G1, G2 or G3 grade columns are used as inputs.
# Direct demographic fields are excluded.
# --------------------------------------------------

FEATURES = [
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "traveltime",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

TARGET_COLUMN = "G3"
PASS_THRESHOLD = 10
RANDOM_STATE = 42


# --------------------------------------------------
# 3. LOAD AND INSPECT THE DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH, sep=";")

required_columns = FEATURES + [TARGET_COLUMN]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

if df.empty:
    raise ValueError("The dataset is empty.")

print("\n========== DATASET INFORMATION ==========")
print("Dataset file:", DATA_PATH.name)
print("Total records:", len(df))
print("Total raw columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))
print("Selected input features:", len(FEATURES))


# --------------------------------------------------
# 4. DEFINE THE PASS / FAIL TARGET
# Pass = G3 >= 10
# Fail = G3 < 10
# --------------------------------------------------

X = df[FEATURES].copy()
y = (df[TARGET_COLUMN] >= PASS_THRESHOLD).astype(int)

print("\nTarget distribution:")
print(
    y.value_counts()
    .rename(index={0: "Fail", 1: "Pass"})
    .to_string()
)


# --------------------------------------------------
# 5. STRATIFIED TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. PREPROCESSING
# Numeric values use median imputation.
# Categorical values use most-frequent imputation
# followed by one-hot encoding.
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# --------------------------------------------------
# 7. RANDOM FOREST CLASSIFIER
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest),
    ]
)


# --------------------------------------------------
# 8. TRAIN THE MODEL
# --------------------------------------------------

print("\n========== MODEL TRAINING ==========")

model.fit(X_train, y_train)

print("Random Forest training completed.")


# --------------------------------------------------
# 9. PREDICT ON UNSEEN TEST DATA
# --------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 10. CALCULATE EVALUATION METRICS
# Positive class = Pass (G3 >= 10)
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)
balanced_accuracy = balanced_accuracy_score(
    y_test, y_pred
)

if y_test.nunique() == 2:
    roc_auc = roc_auc_score(y_test, y_probability)
else:
    roc_auc = None

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    output_dict=True,
    zero_division=0,
)


# --------------------------------------------------
# 11. PRINT RESULTS
# --------------------------------------------------

print("\n========== RANDOM FOREST RESULTS ==========")

print(f"Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC:           {roc_auc:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(cm)

print("\nRows in training set:", len(X_train))
print("Rows in testing set:", len(X_test))


# --------------------------------------------------
# 12. FEATURE IMPORTANCE
# --------------------------------------------------

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_table = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 10 model features:")
print(importance_table.head(10).to_string(index=False))


# --------------------------------------------------
# 13. SAVE EVALUATION ARTIFACTS
# --------------------------------------------------

metrics = {
    "dataset": "UCI Student Performance - Portuguese subset",
    "source": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "dataset_records": int(len(df)),
    "raw_columns": int(len(df.columns)),
    "selected_input_features": FEATURES,
    "selected_feature_count": len(FEATURES),
    "target": "Pass if G3 >= 10; Fail if G3 < 10",
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "model": "RandomForestClassifier",
    "n_estimators": 300,
    "accuracy": float(accuracy),
    "precision_pass_class": float(precision),
    "recall_pass_class": float(recall),
    "f1_pass_class": float(f1),
    "balanced_accuracy": float(balanced_accuracy),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "confusion_matrix_labels": ["Fail", "Pass"],
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "top_10_feature_importance": importance_table.head(10).to_dict(
        orient="records"
    ),
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame(
    cm,
    index=["Actual Fail", "Actual Pass"],
    columns=["Predicted Fail", "Predicted Pass"],
).to_csv(MATRIX_PATH)

model_package = {
    "pipeline": model,
    "feature_columns": FEATURES,
    "target_definition": "Pass if G3 >= 10",
    "dataset_name": "UCI Student Performance - Portuguese subset",
}

joblib.dump(model_package, MODEL_PATH)


# --------------------------------------------------
# 14. FINAL STATUS
# --------------------------------------------------

print("\n========== SAVED FILES ==========")
print("Trained model:", MODEL_PATH)
print("Metrics JSON:", METRICS_PATH)
print("Confusion matrix CSV:", MATRIX_PATH)

print("\nRandom Forest experiment completed successfully.")
