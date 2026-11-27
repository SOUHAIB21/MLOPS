"""Unit tests using synthetic data; no model training or network calls."""
import numpy as np
import pandas as pd
import pytest
from model_pipeline import clean_features, prepare_data


def sample_data():
    return pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=10).strftime("%m/%d/%Y"),
        "quarter": ["Quarter1"] * 10,
        "department": [" sweing "] * 10,
        "day": ["Thursday"] * 10,
        "team": [1] * 10,
        "targeted_productivity": [0.8] * 10,
        "smv": [20.0] * 10,
        "no_of_workers": [30] * 10,
        "actual_productivity": [0.75] * 10,
    })


def test_cleaning():
    data = sample_data()
    data.loc[0, "smv"] = np.inf
    result = clean_features(data)
    assert result.loc[0, "department"] == "sewing"
    assert result.loc[0, "team"] == "1"
    assert pd.isna(result.loc[0, "smv"])
    assert "actual_productivity" not in result.columns
    assert data.loc[0, "department"] == " sweing "


def test_chronological_splits_and_duplicates(tmp_path):
    data = sample_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    splits, quality = prepare_data(path)
    assert quality["duplicate_rows_removed"] == 1
    assert [len(splits[k]) for k in ["train", "validation", "test"]] == [6, 2, 2]
    assert splits["train"]["date"].max() < splits["validation"]["date"].min()
    assert splits["validation"]["date"].max() < splits["test"]["date"].min()


def test_missing_feature_rejected():
    with pytest.raises(ValueError, match="Missing feature columns"):
        clean_features(sample_data().drop(columns="smv"))
