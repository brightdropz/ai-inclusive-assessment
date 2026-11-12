
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "uci_student_performance" / "csv" / "student-por.csv"

FEATURES = [
    "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher",
    "internet", "traveltime", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]

RANDOM_STATE = 42

df = pd.read_csv(DATA, sep=";")

X = df[FEATURES].copy()
y = (df["G3"] >= 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

numeric = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ]), numeric),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

results = []

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Pass Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Pass Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Pass F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Fail Recall": recall_score(
            y_test, predictions, pos_label=0, zero_division=0
        ),
        "ROC-AUC": (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None else None
        ),
    }

    results.append(row)

    print(f"\nCompleted: {name}")

comparison = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(
    index=False,
    formatters={
        column: lambda value: f"{value:.4f}"
        for column in comparison.select_dtypes(
            include="number"
        ).columns
    },
))

output = BASE / "reports" / "model_comparison.csv"
output.parent.mkdir(exist_ok=True)

comparison.to_csv(output, index=False)

print(f"\nComparison saved to: {output}")
