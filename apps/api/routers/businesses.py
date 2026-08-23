from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.auth import get_current_user
from apps.api.deps import get_db
from apps.api.schemas import BusinessOut, CreditRiskOut, FinancialSummaryOut, PeriodParams
from src.config.settings import get_settings
from src.database.models import Business
from src.features.credit_features import get_business_features, load_data
from src.ml.scorecard import get_scorecard
from tools.financial import get_business_profile, get_financial_summary

router = APIRouter(prefix="/businesses", tags=["businesses"])


@router.get("/{business_id}", response_model=BusinessOut)
async def get_business(
    business_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
) -> Business:
    business = db.get(Business, business_id)
    if business is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business not found")
    return business


@router.get("/{business_id}/financial-summary", response_model=FinancialSummaryOut)
async def financial_summary(
    business_id: str,
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
) -> FinancialSummaryOut:
    business = db.get(Business, business_id)
    if business is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business not found")
    summary = get_financial_summary(db, business_id, start_date, end_date)
    return FinancialSummaryOut(**summary)


@router.get("/{business_id}/credit-risk", response_model=CreditRiskOut)
async def credit_risk(
    business_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
) -> CreditRiskOut:
    business = db.get(Business, business_id)
    if business is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business not found")

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synthetic feature data is not available on the server",
        )

    try:
        dfs = load_data(data_dir)
        features = get_business_features(dfs, business_id)
        card = get_scorecard()
        assessment = card.assess(features, top_n=5)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Credit scorecard not available: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return CreditRiskOut(
        business_id=business_id,
        risk_score=assessment["risk_score"],
        probability_of_default=assessment["probability_of_default"],
        risk_level=assessment["risk_level"],
        model_version=assessment["model_version"],
        explanation_drivers=assessment["reason_codes"],
    )
