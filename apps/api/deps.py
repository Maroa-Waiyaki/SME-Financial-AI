from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from src.database.engine import SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
