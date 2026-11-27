import argparse
import subprocess  # nosec B404 - fixed project commands, no shell
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


@task(name="Install dependencies")
def install_dependencies_task():
    subprocess.run(  # nosec B603 - fixed module names and argument lists
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(Path(__file__).resolve().parent / "requirements.txt"),
        ],
        check=True,
    )


@task(name="Prepare data")
def prepare_data_task(csv_path):
    splits, _quality = prepare_data(csv_path)
    return splits


@task(name="Train model")
def train_model_task(splits):
    model, validation_results, _baseline = train_model(
        splits["train"],
        splits["validation"],
    )
    print(validation_results.to_string(index=False))
    return model


@task(name="Save model")
def save_model_task(model, model_path):
    return save_model(model, model_path)


@task(name="Load model")
def load_model_task(model_path):
    return load_model(model_path)


@task(name="Evaluate model")
def evaluate_model_task(model, test_data):
    metrics, _predictions = evaluate_model(model, test_data)
    print(metrics)
    return metrics


# Flow definitions
DATA_PATH = "data/garments_worker_productivity.csv"
MODEL_PATH = "results/random_forest_pipeline.pkl"


PROJECT_DIR = Path(__file__).resolve().parent
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_check(module, *arguments):
    """Run a fixed development tool and stop the flow if it fails."""
    subprocess.run(  # nosec B603 - internal tool names; shell is disabled
        [sys.executable, "-m", module, *arguments],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="Format code")
def format_code_task():
    run_check("black", *CODE_FILES)


@task(name="Check code quality")
def quality_task():
    run_check("ruff", "check", *CODE_FILES)


@task(name="Scan code security")
def security_task():
    run_check("bandit", "-q", "pipeline_prefect.py")


@task(name="Run unit tests")
def unit_tests_task():
    run_check("pytest", "-q", "tests")


@flow(name="code")
def code_flow():
    install_dependencies_task()
    format_code_task()
    quality_task()
    security_task()
    unit_tests_task()


@flow(name="all")
def all_flow():
    code_flow()
    splits = prepare_data_task(DATA_PATH)
    splits["test"].to_csv("results/test.csv", index=False)
    model = train_model_task(splits)
    save_model_task(model, MODEL_PATH)
    evaluate_model_task(model, splits["test"])


@flow(name="train")
def train_flow():
    splits = prepare_data_task(DATA_PATH)
    model = train_model_task(splits)
    save_model_task(model, MODEL_PATH)
    splits["test"].to_csv("results/test.csv", index=False)


@flow(name="evaluate")
def evaluate_flow():
    model = load_model_task(MODEL_PATH)
    test_data = pd.read_csv("results/test.csv")
    evaluate_model_task(model, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code"],
        required=True,
    )
    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    flows = {
        "all": all_flow,
        "code": code_flow,
        "train": train_flow,
        "entrainement": train_flow,
        "evaluate": evaluate_flow,
    }
    flows[args.flow]()
