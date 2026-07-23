from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    value = Settings(
        database_url="sqlite://",
        storage_root=tmp_path / "storage",
        model_cache=tmp_path / "models",
        scan_roots={"test": scan_root},
    )
    value.ensure_directories()
    return value


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
    yield factory
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db(session_factory) -> Generator[Session, None, None]:
    with session_factory() as session:
        yield session


@pytest.fixture
def client(settings: Settings, session_factory) -> Generator[TestClient, None, None]:
    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
