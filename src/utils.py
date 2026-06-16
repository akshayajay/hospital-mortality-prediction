"""
src/utils.py
------------
Utility functions for the in-hospital mortality prediction pipeline.

Extracted from mortality_prediction.ipynb so they can be unit-tested
independently of the notebook runtime.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Columns to drop before modelling — these are identifiers, post-outcome
# variables, or targets that would cause data leakage.
LEAKY_OR_NON_FEATURE_COLS = [
    "id",           # row identifier
    "death",        # ultimate vital status (post-outcome)
    "hospdead",     # target variable
    "slos",         # days from study entry to discharge (post-outcome)
    "d.time",       # days of follow-up (post-outcome)
    "charges",      # hospital charges (post-outcome)
    "totcst",       # total cost (post-outcome)
    "totmcst",      # total micro-cost (post-outcome)
]

TARGET_COL = "hospdead"


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Raise ValueError if the dataframe is missing required columns or is empty.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")


def validate_target(y: pd.Series) -> None:
    """
    Raise ValueError if the target series is not binary (0/1).
    """
    unique = set(y.dropna().unique())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary (0/1). Found values: {unique}"
        )
    if len(unique) < 2:
        raise ValueError(
            "Target column has only one class — cannot train a classifier."
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def drop_leaky_columns(
    df: pd.DataFrame,
    extra_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Drop leaky / non-feature columns from the dataframe.

    Args:
        df:         Input dataframe (features + target).
        extra_cols: Additional column names to drop beyond the defaults.

    Returns:
        DataFrame with leaky columns removed.
    """
    cols_to_drop = list(LEAKY_OR_NON_FEATURE_COLS)
    if extra_cols:
        cols_to_drop += extra_cols
    # Only drop columns that actually exist
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def identify_feature_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Split columns into numeric and categorical lists.

    Args:
        df: Feature DataFrame (no target column).

    Returns:
        (numeric_cols, categorical_cols)
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def get_feature_names(
    preprocessor,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """
    Extract feature names from a fitted ColumnTransformer.

    Args:
        preprocessor:    Fitted sklearn ColumnTransformer.
        numeric_cols:    Original numeric column names.
        categorical_cols: Original categorical column names.

    Returns:
        Flat list of feature names after preprocessing.
    """
    feature_names: List[str] = []

    if "num" in preprocessor.named_transformers_:
        feature_names.extend(numeric_cols)

    if categorical_cols and "cat" in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_["cat"].named_steps.get("onehot")
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names.extend(cat_names)
        else:
            feature_names.extend(categorical_cols)

    return feature_names


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------

def validate_split_sizes(
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    tolerance: int = 5,
) -> None:
    """
    Assert that train + val + test accounts for (roughly) all samples.

    Args:
        n_total:   Total number of samples before splitting.
        n_train:   Training set size.
        n_val:     Validation set size.
        n_test:    Test set size.
        tolerance: Allowed rounding slack in sample counts.
    """
    total_split = n_train + n_val + n_test
    if abs(total_split - n_total) > tolerance:
        raise ValueError(
            f"Split sizes ({n_train} + {n_val} + {n_test} = {total_split}) "
            f"do not sum to n_total ({n_total})."
        )


def compute_class_imbalance_ratio(y: pd.Series) -> float:
    """
    Return the ratio of the majority class to the minority class.

    A ratio > 3 generally warrants class-balancing strategies.
    """
    counts = y.value_counts()
    return float(counts.max() / counts.min())
