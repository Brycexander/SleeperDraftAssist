# syntax=docker/dockerfile:1

FROM python:3.14-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_ARGS="-DSLEEPER_NATIVE_ARCH=x86-64-v2"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md CMakeLists.txt ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels .

FROM python:3.14-slim-bookworm AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    SLEEPER_CACHE_DIR=/app/.cache \
    DRAFT_ENGINE=cpp

RUN groupadd --system draftassist \
    && useradd --system --gid draftassist --home-dir /app draftassist
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels sleeper-draft-assistant \
    && rm -rf /wheels

WORKDIR /app
COPY --chown=draftassist:draftassist .cache /app/.cache
USER draftassist

EXPOSE 8080
CMD ["uvicorn", "sleeper_draft_assistant.web:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
