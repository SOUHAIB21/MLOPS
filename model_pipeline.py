"""Reusable garment productivity regression functions for Workshop 2."""

import hashlib
import json
import platform
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TARGET = "actual_productivity"
CATEGORICAL = ["quarter", "department", "day", "team"]
NUMERIC = ["targeted_productivity", "smv", "no_of_workers"]
FEATURES = CATEGORICAL + NUMERIC
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/597/"
    "productivity%2Bprediction%2Bof%2Bgarment%2Bemployees.zip"
)


def download_data(path):
    """Download the UCI CSV if absent; preserve an existing local dataset."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            DATA_URL, timeout=120
        ) as response:  # nosec B310 - fixed HTTPS UCI URL
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            member = next(
                n
                for n in archive.namelist()
                if n.endswith("garments_worker_productivity.csv")
            )
            path.write_bytes(archive.read(member))
    return path


def clean_features(frame):
    """Apply deterministic cleanup; leave numeric imputation to the pipeline."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    result = frame[FEATURES].copy()
    for column in ["quarter", "department", "day"]:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
            .fillna("unknown")
            .astype(object)
        )
    result["department"] = result["department"].replace({"sweing": "sewing"})
    team = pd.to_numeric(result["team"], errors="coerce")
    result["team"] = team.map(
        lambda x: "unknown" if not np.isfinite(x) else format(x, "g")
    )
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result[NUMERIC] = result[NUMERIC].replace([np.inf, -np.inf], np.nan)
    return result


def prepare_data(csv_path):
    """Load, validate and split raw data by date into 60/20/20 date groups.

    Returns (splits, quality). Learned preprocessing is deferred to training,
    preventing information from validation/test data entering the imputers.
    """
    raw = pd.read_csv(csv_path)
    required = set(FEATURES + ["date", TARGET])
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    data = raw.drop_duplicates().copy()
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    valid = data["date"].notna() & np.isfinite(data[TARGET])
    invalid_count = int((~valid).sum())
    data = data.loc[valid].sort_values("date").reset_index(drop=True)
    dates = np.sort(data["date"].unique())
    if len(dates) < 10:
        raise ValueError("At least 10 valid distinct dates are required.")
    a, b = int(len(dates) * 0.6), int(len(dates) * 0.8)
    splits = {
        name: data.loc[data["date"].isin(group)].copy()
        for name, group in [
            ("train", dates[:a]),
            ("validation", dates[a:b]),
            ("test", dates[b:]),
        ]
    }
    quality = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": int(raw.duplicated().sum()),
        "invalid_date_or_target_rows_removed": invalid_count,
        "target_above_one_retained": int(data[TARGET].gt(1).sum()),
        "missing_values_raw": raw.isna().sum().to_dict(),
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "split": {
            name: {
                "rows": len(part),
                "dates": part["date"].nunique(),
                "first_date": str(part["date"].min()),
                "last_date": str(part["date"].max()),
            }
            for name, part in splits.items()
        },
    }
    return splits, quality


def build_pipeline(estimator):
    """Construct an unfitted pipeline including cleanup and encoding."""
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                NUMERIC,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_features, validate=False)),
            ("preprocess", preprocessing),
            ("regressor", estimator),
        ]
    )


def evaluate_model(model, data):
    """Return regression metrics and predictions without changing the model."""
    if data.empty or TARGET not in data:
        raise ValueError("Evaluation requires nonempty data with the target column.")
    actual = data[TARGET]
    predicted = model.predict(data[FEATURES])
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }, predicted


def train_model(train, validation, seed=42, n_estimators=200):
    """Select forest settings on validation MAE and refit on train+validation.

    Returns (fitted_forest_pipeline, validation_table, fitted_mean_baseline).
    Test data is deliberately not accepted by this function.
    """
    if train.empty or validation.empty:
        raise ValueError("Training and validation splits must be nonempty.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive.")
    candidates = {"Mean baseline": DummyRegressor(strategy="mean")}
    for depth in [8, None]:
        for leaf in [2, 5]:
            candidates[f"Random Forest depth={depth}, leaf={leaf}"] = (
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=0.8,
                    random_state=seed,
                    n_jobs=2,
                )
            )
    fitted, rows = {}, []
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(train[FEATURES], train[TARGET])
        metrics, _ = evaluate_model(pipeline, validation)
        fitted[name] = pipeline
        rows.append({"model": name, **metrics})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    best = table.loc[table["model"].str.startswith("Random Forest"), "model"].iloc[0]
    development = pd.concat([train, validation], ignore_index=True)
    final = clone(fitted[best]).fit(development[FEATURES], development[TARGET])
    baseline = clone(fitted["Mean baseline"]).fit(
        development[FEATURES], development[TARGET]
    )
    return final, table, baseline


def save_model(model, path):
    """Save the entire pipeline using joblib, with a .pkl extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    """Load a trusted joblib PKL; keep model_pipeline.py importable."""
    return joblib.load(path)  # nosec B301 - only load trusted project artifacts


def save_metadata(path, model, seed):
    """Record the selected forest settings, feature schema and runtime versions."""
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "seed": seed,
        "selection": "Lowest validation MAE among four random forest configurations",
        "forest_parameters": model.named_steps["regressor"].get_params(),
        "serialization": "joblib; model_pipeline.py must be importable",
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
    }
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
