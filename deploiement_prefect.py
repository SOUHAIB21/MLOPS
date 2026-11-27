"""Register and serve workshop deployments. Run explicitly in terminal 2."""
import os
from pathlib import Path

from prefect import serve
from prefect.schedules import Cron
from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow


def main():
    # Existing workflow paths are relative to the project directory.
    os.chdir(Path(__file__).resolve().parent)
    deployments = [
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 9 * * *", timezone="Africa/Lagos"),
            description="Full workflow daily at 09:00 Africa/Lagos.",
        ),
        train_flow.to_deployment(name="ml-pipeline-train"),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate"),
        code_flow.to_deployment(name="ml-pipeline-code"),
    ]
    # Run at most one workflow at a time to avoid shared-file collisions.
    serve(*deployments, limit=1, pause_on_shutdown=True)


if __name__ == "__main__":
    main()
