"""CLI orchestration. Importing this file does not train or download anything."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_pipeline import (
    FEATURES,
    TARGET,
    download_data,
    evaluate_model,
    load_model,
    prepare_data,
    save_metadata,
    save_model,
    train_model,
)


def main():
    """Run the requested step; use --step all for the full workflow."""
    parser = argparse.ArgumentParser(
        description="Workshop 2: garment random forest pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["all", "prepare", "train", "evaluate", "predict"],
        default="all",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/garments_worker_productivity.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--download", action="store_true", help="Download UCI data if absent"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--input", type=Path, help="CSV of raw feature rows for --step predict"
    )
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "random_forest_pipeline.pkl"

    if args.step == "predict":
        if args.input is None:
            parser.error("--step predict requires --input features.csv")
        frame = pd.read_csv(args.input)
        frame["predicted_productivity"] = load_model(model_path).predict(frame)
        frame.to_csv(root / "predictions.csv", index=False)
        print("Saved predictions:", root / "predictions.csv")
        return

    if args.step in ["all", "prepare"]:
        if args.download:
            download_data(args.data)
        splits, quality = prepare_data(args.data)
        for name, part in splits.items():
            part.to_csv(root / f"{name}.csv", index=False)
        (root / "data_quality.json").write_text(
            json.dumps(quality, indent=2), encoding="utf-8"
        )
        print("Prepared chronological splits:", {k: len(v) for k, v in splits.items()})
    if args.step in ["all", "train"]:
        train = pd.read_csv(root / "train.csv")
        validation = pd.read_csv(root / "validation.csv")
        model, table, baseline = train_model(
            train, validation, args.seed, args.n_estimators
        )
        table.to_csv(root / "validation_metrics.csv", index=False)
        save_model(model, model_path)
        save_model(baseline, root / "mean_baseline.pkl")
        save_metadata(root / "model_metadata.json", model, args.seed)
        print(table.to_string(index=False))
        print("Saved complete pipeline:", model_path)
    if args.step in ["all", "evaluate"]:
        test = pd.read_csv(root / "test.csv")
        reloaded = load_model(model_path)
        metrics, predictions = evaluate_model(reloaded, test)
        baseline_metrics, _ = evaluate_model(
            load_model(root / "mean_baseline.pkl"), test
        )
        comparison = pd.DataFrame(
            [metrics, baseline_metrics], index=["Random Forest", "Mean baseline"]
        )
        comparison.to_csv(root / "test_metrics.csv")
        report = test[["date", "department", "team", TARGET]].copy()
        report["prediction"] = predictions
        report["residual"] = report[TARGET] - predictions
        report.to_csv(root / "test_predictions.csv", index=False)
        if args.step == "all":
            np.testing.assert_allclose(model.predict(test[FEATURES]), predictions)
            print("Saved/reloaded model predictions match.")
        print(comparison.to_string())


if __name__ == "__main__":
    main()
