"""
train_compare.py
-----------------
Trains and tunes all 5 candidate algorithms (KNN, Decision Tree, Naive Bayes,
Random Forest, Logistic Regression) on the same train/test split, and prints
a single ranked comparison table.

This is the "which model wins" script. It does NOT save a deployable pipeline
— that happens in train_final.py, once you've picked a model based on the
output of this script.

Run:
    python src/train_compare.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_and_prepare

DATA_PATH = "../data/neo_messy_balanced.csv"
RANDOM_STATE = 42


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }


def tune(estimator, param_grid, X_train, y_train, scoring="f1"):
    grid = GridSearchCV(estimator, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"  Best Parameters: {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")
    return grid.best_estimator_


def main():
    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    print("\n=== KNN ===")
    knn_grid = {
        "n_neighbors": range(1, 31),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    best_knn = tune(KNeighborsClassifier(), knn_grid, X_train_scaled, y_train)
    results["KNN"] = evaluate(best_knn, X_test_scaled, y_test)

    print("\n=== Decision Tree ===")
    dt_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    }
    best_dt = tune(DecisionTreeClassifier(random_state=RANDOM_STATE), dt_grid, X_train_scaled, y_train)
    results["Decision Tree"] = evaluate(best_dt, X_test_scaled, y_test)

    print("\n=== Naive Bayes ===")
    nb_grid = {"var_smoothing": np.logspace(0, -9, num=100)}
    best_nb = tune(GaussianNB(), nb_grid, X_train_scaled, y_train)
    results["Naive Bayes"] = evaluate(best_nb, X_test_scaled, y_test)

    print("\n=== Random Forest ===")
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    best_rf = tune(RandomForestClassifier(random_state=RANDOM_STATE), rf_grid, X_train_scaled, y_train)
    results["Random Forest"] = evaluate(best_rf, X_test_scaled, y_test)

    print("\n=== Logistic Regression ===")
    lr_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    best_lr = tune(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), lr_grid, X_train_scaled, y_train)
    results["Logistic Regression"] = evaluate(best_lr, X_test_scaled, y_test)

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values("F1 Score", ascending=False)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (sorted by F1 Score)")
    print("=" * 60)
    print(comparison_df.round(4))

    comparison_df.to_csv("../models/model_comparison_results.csv")
    print("\nSaved results to models/model_comparison_results.csv")


if __name__ == "__main__":
    main()
