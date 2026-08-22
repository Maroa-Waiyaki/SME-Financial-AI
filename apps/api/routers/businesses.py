from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from apps.api.auth import get_current_user
from apps.api.deps import get_db
from apps.api.schemas import BusinessOut, FinancialSummaryOut, PeriodParams
from src.database.models import Business
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
