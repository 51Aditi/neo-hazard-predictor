"""
preprocessing.py
-----------------
All data cleaning / preparation logic for the NEO hazard dataset lives here,
so train_compare.py and train_final.py (and anything else) stay in sync and
never duplicate cleaning logic.

NOTE: Scaling is NOT done here. Scaling is a model-dependent step and is
handled inside the sklearn Pipeline in train_final.py, so it's saved
*with* the model and can never go out of sync at inference time.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


DROP_COLS = ["id", "orbiting_body", "sentry_object", "name"]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV as-is."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps from the EDA notebook:
    - clip relative_velocity outliers (IQR upper bound)
    - fix negative miss_distance values
    - fill missing values
    - drop duplicates
    - drop non-predictive / identifier columns
    - label-encode the target
    Returns a clean, model-ready dataframe (target still included as 'hazardous').
    """
    df = df.copy()

    # --- outlier handling ---
    q1 = df["relative_velocity"].quantile(0.25)
    q3 = df["relative_velocity"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df["relative_velocity"] = df["relative_velocity"].clip(upper=upper_bound)

    df["miss_distance"] = df["miss_distance"].abs()

    # est_diameter_min/max, absolute_magnitude outliers are left untouched —
    # they reflect real astronomical size/brightness distributions, not errors.

    # --- missing values ---
    df["est_diameter_min"] = df["est_diameter_min"].fillna(df["est_diameter_min"].median())
    df["relative_velocity"] = df["relative_velocity"].fillna(df["relative_velocity"].mean())
    df["miss_distance"] = df["miss_distance"].fillna(df["miss_distance"].mean())
    df["absolute_magnitude"] = df["absolute_magnitude"].fillna(df["absolute_magnitude"].median())

    # --- duplicates ---
    df = df.drop_duplicates()

    # --- drop non-predictive columns ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # --- encode target ---
    le = LabelEncoder()
    df["hazardous"] = le.fit_transform(df["hazardous"])

    return df


def get_features_target(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (target)."""
    X = df.drop(columns=["hazardous"])
    y = df["hazardous"]
    return X, y


def load_and_prepare(csv_path: str):
    """Convenience: load raw csv -> clean -> return X, y."""
    df = load_raw(csv_path)
    df = clean_data(df)
    return get_features_target(df)
