from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings
from src.database.base import Base


def make_engine(database_url: str | None = None, pool_size: int = 10, max_overflow: int = 20):
    if database_url is None:
        database_url = str(get_settings().database_url)
    return create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        future=True,
    )


def get_engine():
    return make_engine()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
