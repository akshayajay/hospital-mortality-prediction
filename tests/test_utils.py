"""
tests/test_utils.py
-------------------
Unit tests for the mortality prediction utility functions in src/utils.py.

No data download, no model training, no network calls required.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    validate_dataframe,
    validate_target,
    drop_leaky_columns,
    identify_feature_types,
    get_feature_names,
    validate_split_sizes,
    compute_class_imbalance_ratio,
    LEAKY_OR_NON_FEATURE_COLS,
    TARGET_COL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal SUPPORT2-like DataFrame with leaky and feature columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "id":       range(n),
        "hospdead": np.random.randint(0, 2, n),
        "death":    np.random.randint(0, 2, n),
        "age":      np.random.randint(18, 90, n).astype(float),
        "meanbp":   np.random.uniform(40, 120, n),
        "sex":      np.random.choice(["male", "female"], n),
        "race":     np.random.choice(["white", "black", "other"], n),
        "slos":     np.random.randint(1, 30, n).astype(float),
        "charges":  np.random.uniform(1000, 50000, n),
    })


@pytest.fixture
def clean_features(sample_df):
    """Feature-only DataFrame with leaky columns removed."""
    return drop_leaky_columns(sample_df)


@pytest.fixture
def binary_target():
    return pd.Series([0, 1, 0, 0, 1, 1, 0, 1], name="hospdead")


# ---------------------------------------------------------------------------
# LEAKY_OR_NON_FEATURE_COLS
# ---------------------------------------------------------------------------

class TestConstants:
    def test_target_col_is_hospdead(self):
        assert TARGET_COL == "hospdead"

    def test_leaky_cols_includes_target(self):
        assert "hospdead" in LEAKY_OR_NON_FEATURE_COLS

    def test_leaky_cols_includes_id(self):
        assert "id" in LEAKY_OR_NON_FEATURE_COLS

    def test_leaky_cols_includes_death(self):
        assert "death" in LEAKY_OR_NON_FEATURE_COLS

    def test_leaky_cols_no_duplicates(self):
        assert len(LEAKY_OR_NON_FEATURE_COLS) == len(set(LEAKY_OR_NON_FEATURE_COLS))


# ---------------------------------------------------------------------------
# validate_dataframe()
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_valid_df_passes(self, sample_df):
        validate_dataframe(sample_df)  # should not raise

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_dataframe(None)

    def test_missing_target_col_raises(self, sample_df):
        df = sample_df.drop(columns=["hospdead"])
        with pytest.raises(ValueError, match="hospdead"):
            validate_dataframe(df)


# ---------------------------------------------------------------------------
# validate_target()
# ---------------------------------------------------------------------------

class TestValidateTarget:
    def test_valid_binary_passes(self, binary_target):
        validate_target(binary_target)

    def test_non_binary_raises(self):
        y = pd.Series([0, 1, 2, 3])
        with pytest.raises(ValueError, match="binary"):
            validate_target(y)

    def test_single_class_raises(self):
        y = pd.Series([0, 0, 0, 0])
        with pytest.raises(ValueError, match="one class"):
            validate_target(y)

    def test_nan_values_ignored(self):
        y = pd.Series([0, 1, np.nan, 0, 1])
        validate_target(y)  # should not raise

    def test_float_binary_passes(self):
        y = pd.Series([0.0, 1.0, 0.0, 1.0])
        validate_target(y)


# ---------------------------------------------------------------------------
# drop_leaky_columns()
# ---------------------------------------------------------------------------

class TestDropLeakyColumns:
    def test_removes_id(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "id" not in result.columns

    def test_removes_target(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "hospdead" not in result.columns

    def test_removes_death(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "death" not in result.columns

    def test_removes_charges(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "charges" not in result.columns

    def test_keeps_age(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "age" in result.columns

    def test_keeps_meanbp(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert "meanbp" in result.columns

    def test_extra_cols_removed(self, sample_df):
        result = drop_leaky_columns(sample_df, extra_cols=["sex"])
        assert "sex" not in result.columns

    def test_missing_leaky_col_no_error(self):
        """Should not raise if a leaky column isn't present in the df."""
        df = pd.DataFrame({"age": [30, 40], "hospdead": [0, 1]})
        result = drop_leaky_columns(df)
        assert "age" in result.columns

    def test_returns_dataframe(self, sample_df):
        result = drop_leaky_columns(sample_df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# identify_feature_types()
# ---------------------------------------------------------------------------

class TestIdentifyFeatureTypes:
    def test_separates_numeric_and_categorical(self, clean_features):
        numeric, categorical = identify_feature_types(clean_features)
        assert "age" in numeric
        assert "meanbp" in numeric
        assert "sex" in categorical
        assert "race" in categorical

    def test_numeric_not_in_categorical(self, clean_features):
        numeric, categorical = identify_feature_types(clean_features)
        overlap = set(numeric) & set(categorical)
        assert len(overlap) == 0

    def test_all_columns_accounted_for(self, clean_features):
        numeric, categorical = identify_feature_types(clean_features)
        assert set(numeric) | set(categorical) == set(clean_features.columns)

    def test_all_numeric_df(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        numeric, categorical = identify_feature_types(df)
        assert set(numeric) == {"a", "b"}
        assert categorical == []

    def test_all_categorical_df(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        numeric, categorical = identify_feature_types(df)
        assert numeric == []
        assert set(categorical) == {"x", "y"}


# ---------------------------------------------------------------------------
# get_feature_names()
# ---------------------------------------------------------------------------

class TestGetFeatureNames:
    def _make_preprocessor(self, numeric_cols, categorical_cols, cat_out_names):
        ohe = MagicMock()
        ohe.get_feature_names_out.return_value = np.array(cat_out_names)

        cat_transformer = MagicMock()
        cat_transformer.named_steps = {"onehot": ohe}

        preprocessor = MagicMock()
        preprocessor.named_transformers_ = {
            "num": MagicMock(),
            "cat": cat_transformer,
        }
        return preprocessor

    def test_includes_numeric_names(self):
        numeric = ["age", "meanbp"]
        cat_out = ["sex_female", "sex_male"]
        prep = self._make_preprocessor(numeric, ["sex"], cat_out)
        result = get_feature_names(prep, numeric, ["sex"])
        assert "age" in result
        assert "meanbp" in result

    def test_includes_ohe_names(self):
        numeric = ["age"]
        cat_out = ["race_black", "race_other", "race_white"]
        prep = self._make_preprocessor(numeric, ["race"], cat_out)
        result = get_feature_names(prep, numeric, ["race"])
        assert "race_black" in result
        assert "race_white" in result

    def test_numeric_before_categorical(self):
        numeric = ["age", "meanbp"]
        cat_out = ["sex_female", "sex_male"]
        prep = self._make_preprocessor(numeric, ["sex"], cat_out)
        result = get_feature_names(prep, numeric, ["sex"])
        assert result.index("age") < result.index("sex_female")

    def test_no_categorical_returns_only_numeric(self):
        numeric = ["age", "meanbp"]
        preprocessor = MagicMock()
        preprocessor.named_transformers_ = {"num": MagicMock()}
        result = get_feature_names(preprocessor, numeric, [])
        assert result == numeric


# ---------------------------------------------------------------------------
# validate_split_sizes()
# ---------------------------------------------------------------------------

class TestValidateSplitSizes:
    def test_exact_split_passes(self):
        validate_split_sizes(100, 60, 20, 20)

    def test_within_tolerance_passes(self):
        validate_split_sizes(100, 60, 20, 19, tolerance=5)

    def test_exceeds_tolerance_raises(self):
        with pytest.raises(ValueError, match="sum"):
            validate_split_sizes(100, 60, 20, 10, tolerance=5)

    def test_zero_tolerance_exact_match(self):
        validate_split_sizes(500, 300, 100, 100, tolerance=0)


# ---------------------------------------------------------------------------
# compute_class_imbalance_ratio()
# ---------------------------------------------------------------------------

class TestComputeClassImbalanceRatio:
    def test_balanced_returns_one(self):
        y = pd.Series([0, 1, 0, 1, 0, 1])
        assert compute_class_imbalance_ratio(y) == pytest.approx(1.0)

    def test_imbalanced_returns_correct_ratio(self):
        y = pd.Series([0] * 90 + [1] * 10)
        assert compute_class_imbalance_ratio(y) == pytest.approx(9.0)

    def test_returns_float(self):
        y = pd.Series([0, 0, 0, 1])
        result = compute_class_imbalance_ratio(y)
        assert isinstance(result, float)

    def test_support2_like_imbalance(self):
        """SUPPORT2 has ~20% mortality — ratio should be around 4."""
        y = pd.Series([0] * 80 + [1] * 20)
        ratio = compute_class_imbalance_ratio(y)
        assert ratio == pytest.approx(4.0)
