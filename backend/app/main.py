from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import assets, health, jobs
from app.api.errors import install_error_handlers
from app.config import get_settings
from app.db.init_db import migrate_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    settings = get_settings()
    settings.ensure_directories()
    migrate_database()
    yield


app = FastAPI(
    title="Action Recognition Local API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_error_handlers(app)
app.include_router(health.router, prefix="/api")
app.include_router(assets.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
