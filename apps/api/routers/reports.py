from __future__ import annotations

import io
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from apps.api.auth import get_current_user
from apps.api.deps import get_db
from tools.financial_reports import (
    export_financial_excel,
    export_financial_pptx,
    generate_balance_sheet,
    generate_pnl_statement,
)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/pnl")
async def get_pnl(
    business_id: str = Query(..., description="Business ID (e.g. B000001)"),
    start_date: str = Query("2023-01-01"),
    end_date: str = Query("2023-12-31"),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    try:
        return generate_pnl_statement(db, business_id, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/balance-sheet")
async def get_balance_sheet(
    business_id: str = Query(..., description="Business ID (e.g. B000001)"),
    as_of_date: str = Query("2023-12-31"),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    try:
        return generate_balance_sheet(db, business_id, as_of_date)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/export/pptx")
async def export_pptx(
    business_id: str = Query(..., description="Business ID (e.g. B000001)"),
    start_date: str = Query("2023-01-01"),
    end_date: str = Query("2023-12-31"),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    try:
        pnl = generate_pnl_statement(db, business_id, start_date, end_date)
        bs = generate_balance_sheet(db, business_id, end_date)
        content = export_financial_pptx(pnl, bs)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{business_id}_Financial_Review.pptx"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/export/xlsx")
async def export_xlsx(
    business_id: str = Query(..., description="Business ID (e.g. B000001)"),
    start_date: str = Query("2023-01-01"),
    end_date: str = Query("2023-12-31"),
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    try:
        pnl = generate_pnl_statement(db, business_id, start_date, end_date)
        bs = generate_balance_sheet(db, business_id, end_date)
        content = export_financial_excel(pnl, bs)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{business_id}_Financial_Model.xlsx"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
