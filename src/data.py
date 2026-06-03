from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# The dataset target column that tells us whether an employee left.
RAW_TARGET = "Attrition"

# Columns that are identifiers or constants, so they do not help prediction.
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    # Read the CSV from disk into a pandas DataFrame.
    return pd.read_csv(Path(path))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy so the original dataframe is not modified in place.
    frame = df.copy()

    # Convert the target label from text to 0/1 so ML models can learn from it.
    frame[RAW_TARGET] = frame[RAW_TARGET].map({"Yes": 1, "No": 0}).astype(int)

    # Convert simple binary features into integers to keep the pipeline numeric.
    frame["Gender"] = frame["Gender"].map({"Male": 1, "Female": 0}).astype(int)
    frame["Over18"] = frame["Over18"].map({"Y": 1}).astype(int)
    frame["OverTime"] = frame["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

    # Remove columns that are constant or only identify a row.
    frame = frame.drop(columns=DROP_COLUMNS)
    return frame


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Separate target from the feature matrix.
    y = df[RAW_TARGET].astype(int)
    X = df.drop(columns=[RAW_TARGET])
    return X, y


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    # Object columns are treated as categorical, everything else as numeric.
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numerical = [c for c in X.columns if c not in categorical]
    return numerical, categorical
