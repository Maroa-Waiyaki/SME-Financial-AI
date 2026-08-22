from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import mlflow

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def configure_mlflow() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)


class MLflowTracker(contextlib.AbstractContextManager):
    def __init__(self, run_name: str | None = None) -> None:
        self.run_name = run_name
        self.run = None

    def __enter__(self) -> "MLflowTracker":
        configure_mlflow()
        self.run = mlflow.start_run(run_name=self.run_name)
        logger.info(f"MLflow run started: {self.run.info.run_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val:
            mlflow.log_param("error", str(exc_val))
        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(model, artifact_path)

    def register_model(self, model_name: str, artifact_path: str = "model") -> None:
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mlflow.register_model(model_uri, model_name)
