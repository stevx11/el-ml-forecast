"""Health check responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthOut(BaseModel):
    """Liveness. Deliberately has no dependencies -- it answers "is the process up?"."""

    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str


class ReadinessOut(BaseModel):
    """Readiness. What an orchestrator should poll before routing traffic.

    ``redis`` is reported but does not gate readiness: the cache is optional and
    the API serves correctly without it, so failing readiness on a Redis outage
    would take the service down for no reason.
    """

    status: Literal["ready", "degraded"]
    database: bool
    redis: bool
    detail: str | None = None
