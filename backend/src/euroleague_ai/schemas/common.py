"""Shared response envelopes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageMeta(BaseModel):
    """Pagination metadata."""

    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0)
    returned: int = Field(ge=0)


class Page[T](BaseModel):
    """A page of results.

    Always an envelope, never a bare array: adding pagination metadata to a
    top-level JSON array later is a breaking change, and this costs nothing now.
    """

    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    meta: PageMeta
