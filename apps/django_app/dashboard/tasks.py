"""Celery tasks for proactive SME monitoring and risk reassessment."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from celery import shared_task

from src.config.settings import get_settings
from src.database.engine import SessionLocal
from src.database.models import Alert, MonitoringRun, ModelPrediction
from src.features.credit_features import build_features, load_data
from src.ml.scorecard import get_scorecard

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def run_credit_monitoring(self) -> dict:
    """Re-assess credit risk for every business and create alerts for high-risk cases."""
    run_id = str(uuid.uuid4())
    run = MonitoringRun(
        run_id=run_id,
        started_at=_now(),
        businesses_scanned=0,
        alerts_created=0,
        status="running",
    )

    with SessionLocal() as session:
        session.add(run)
        session.commit()

        settings = get_settings()
        data_dir = Path(settings.data_dir)
        if not data_dir.exists():
            run.status = "failed"
            run.finished_at = _now()
            run.detail = {"error": f"Data directory {data_dir} not found"}
            session.commit()
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        try:
            dfs = load_data(data_dir)
            features = build_features(dfs)
            scorecard = get_scorecard()
        except FileNotFoundError as exc:
            run.status = "failed"
            run.finished_at = _now()
            run.detail = {"error": str(exc)}
            session.commit()
            raise self.retry(exc=exc)

        alerts_created = 0
        businesses_scanned = 0
        for _, row in features.iterrows():
            business_id = str(row["business_id"])
            businesses_scanned += 1
            feature_dict = row.to_dict()
            assessment = scorecard.assess(feature_dict, top_n=5)

            pred = ModelPrediction(
                prediction_id=str(uuid.uuid4()),
                business_id=business_id,
                created_at=_now(),
                model_name=assessment["model_name"],
                model_version=assessment["model_version"],
                probability_of_default=assessment["probability_of_default"],
                risk_score=assessment["risk_score"],
                risk_level=assessment["risk_level"],
                features=feature_dict,
                reason_codes=assessment["reason_codes"],
            )
            session.add(pred)

            if assessment["risk_level"] == "high":
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    business_id=business_id,
                    created_at=_now(),
                    alert_type="credit_risk_high",
                    severity="high",
                    title="High credit-risk detected",
                    description=(
                        f"{business_id} has a risk score of {assessment['risk_score']} "
                        f"(probability of default: {assessment['probability_of_default']})."
                    ),
                    recommended_action="review",
                    evidence={
                        "risk_score": assessment["risk_score"],
                        "probability_of_default": assessment["probability_of_default"],
                        "reason_codes": assessment["reason_codes"],
                    },
                    source="proactive_monitoring",
                )
                session.add(alert)
                alerts_created += 1

        run.businesses_scanned = businesses_scanned
        run.alerts_created = alerts_created
        run.status = "completed"
        run.finished_at = _now()
        session.commit()

    return {
        "run_id": run_id,
        "businesses_scanned": businesses_scanned,
        "alerts_created": alerts_created,
    }
