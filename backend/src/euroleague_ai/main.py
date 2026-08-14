"""FastAPI application factory."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from euroleague_ai import __version__
from euroleague_ai.api.router import api_router
from euroleague_ai.api.v1 import health
from euroleague_ai.core.config import Settings, get_settings
from euroleague_ai.core.errors import EuroLeagueAIError, NotFoundError, ProviderError
from euroleague_ai.core.logging import configure_logging, get_logger, set_correlation_id
from euroleague_ai.db.session import dispose_engine, init_engine

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the engine at startup, dispose it at shutdown.

    Deliberately does *not* verify the database connects. A database outage
    should surface on ``/health/ready`` and be recoverable, not prevent the
    process from starting at all.
    """
    settings: Settings = app.state.settings
    init_engine(settings)
    logger.info(
        "application started",
        extra={"version": __version__, "environment": str(settings.environment)},
    )
    try:
        yield
    finally:
        await dispose_engine()
        logger.info("application stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can construct an app
    with overridden settings without mutating global state.
    """
    resolved = settings or get_settings()
    configure_logging(level=resolved.log_level, fmt=resolved.log_format)

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        description=(
            "Pre-game EuroLeague intelligence and player-performance prediction. "
            "Read endpoints cover the ingested domain data; prediction endpoints "
            "appear once models exist."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = resolved

    _add_middleware(app, resolved)
    _add_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(api_router, prefix=resolved.api_v1_prefix)
    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the web client from the API container, if it is present.

    No build step and no second service. The client is three static files, so
    handing them to the process that already serves the data removes a whole
    deployment: no node image, no bundler, no CORS between them, and nothing
    that can drift out of sync with the API it talks to.

    Mounted last so it cannot shadow ``/api``, ``/docs`` or ``/health``, and
    skipped silently when the directory is absent -- an API-only deployment is a
    perfectly reasonable way to run this.
    """
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    # Two candidates: the repo layout (frontend/ beside backend/) and the
    # container layout, where it is copied to /app/frontend.
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "frontend",  # backend/src/euroleague_ai/main.py -> repo root
        Path("/app/frontend"),
    ]
    for directory in candidates:
        if (directory / "index.html").is_file():
            app.mount(
                "/app",
                StaticFiles(directory=str(directory), html=True),
                name="frontend",
            )
            logger.info("frontend mounted", extra={"path": str(directory)})
            return
    logger.info("no frontend directory found; serving API only")


def _add_middleware(app: FastAPI, settings: Settings) -> None:
    if not settings.is_production:
        # Permissive only outside production, so the future Angular dev server
        # can call the API without a proxy. Production origins must be explicit.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def correlation_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Bind a correlation id and log request timing.

        The id is taken from the inbound header when present so a trace survives
        across services, and echoed back so a client can quote it in a bug report.
        """
        correlation_id = set_correlation_id(request.headers.get(CORRELATION_HEADER))
        started = time.perf_counter()

        response = await call_next(request)

        response.headers[CORRELATION_HEADER] = correlation_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response


def _add_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def handle_not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def handle_provider_error(_: Request, exc: ProviderError) -> JSONResponse:
        # 502: the failure is upstream, not in the client's request.
        logger.warning("provider error", extra={"error": str(exc)})
        return JSONResponse(status_code=502, content={"detail": "upstream data provider failed"})

    @app.exception_handler(EuroLeagueAIError)
    async def handle_application_error(_: Request, exc: EuroLeagueAIError) -> JSONResponse:
        # The message is logged but not returned: internal errors can carry
        # detail that should not reach a client.
        logger.exception("unhandled application error", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "internal error"})


app = create_app()
