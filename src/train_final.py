"""
train_final.py
---------------
Builds and saves the ONE final deployable pipeline: scaling + model bundled
together in a single sklearn Pipeline object. This is the artifact the
Streamlit app and any other consumer loads — nobody downstream needs to
remember to scale inputs separately, because it's baked into the pipeline.

Set FINAL_MODEL below to whichever model won in train_compare.py.

Run:
    python src/train_final.py
"""

import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing import load_and_prepare

DATA_PATH = "../data/neo_messy_balanced.csv"
MODEL_OUT_PATH = "../models/neo_hazard_pipeline.pkl"
FEATURE_LIST_PATH = "../models/feature_columns.json"
RANDOM_STATE = 42

# --- Set this to whatever won train_compare.py ---
# Swap this block out for RandomForestClassifier(...) / LogisticRegression(...)
# etc. with its best_params_ once you've decided on a winner.
FINAL_MODEL = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
)


def main():
    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scaler + model bundled into ONE pipeline -> saved together, always in sync.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", FINAL_MODEL),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("Final Pipeline Performance")
    print("-" * 30)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved pipeline to {MODEL_OUT_PATH}")

    # Save the exact feature column order/names — the Streamlit app needs
    # this to build inputs in the right order without guessing.
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"Saved feature column list to {FEATURE_LIST_PATH}")

    # Feature importance, if the model supports it
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        print("\nFeature Importance:")
        print(importance_df)


if __name__ == "__main__":
    main()
