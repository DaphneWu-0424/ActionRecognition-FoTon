from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorBody
