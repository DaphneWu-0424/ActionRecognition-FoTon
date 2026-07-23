from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


settings = get_settings()
engine = create_engine(
    settings.database_url,
    connect_args=(
        {"check_same_thread": False, "timeout": 30}
        if settings.database_url.startswith("sqlite")
        else {}
    ),
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(Engine, "connect")
    def configure_sqlite(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
