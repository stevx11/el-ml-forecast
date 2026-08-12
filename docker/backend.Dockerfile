# syntax=docker/dockerfile:1
#
# Two stages so the runtime image carries no build toolchain and no compilers.
FROM python:3.12-slim AS builder

# Dev tooling (pytest, ruff, mypy) is installed by default because this image
# is the development environment. Build with --build-arg INSTALL_EXTRAS="" for
# a lean production image.
ARG INSTALL_EXTRAS="[dev]"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only what the build backend needs to resolve dependencies, so a source
# edit does not invalidate the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY backend/src ./backend/src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".${INSTALL_EXTRAS}"


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/backend/src"

# libpq is needed at runtime by asyncpg's fallback paths and by psql-style tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./
COPY --chown=appuser:appuser backend ./backend

# Never run as root.
USER appuser

EXPOSE 8000

CMD ["uvicorn", "euroleague_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
