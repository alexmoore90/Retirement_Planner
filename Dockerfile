# ── Stage 1: dependency resolution ───────────────────────────────────────────
# Use uv's official image for the build stage — fast, reproducible
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy lockfile and project metadata first (layer-cache friendly)
# uv sync --frozen installs exactly what's in uv.lock — no resolution needed
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv, no dev extras
# --no-install-project: don't install the package itself yet (src not copied)
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash planner
WORKDIR /home/planner/app

# Copy the pre-built venv from builder — no uv needed at runtime
COPY --from=builder /app/.venv /home/planner/app/.venv

# Copy application source
COPY retirement_model.py ./

# Optional: copy HTML planner (served separately or as static asset)
COPY retirement_planner_final.html ./

# Activate venv by prepending to PATH
ENV PATH="/home/planner/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cache directory for market data — mount as volume in production
RUN mkdir -p /home/planner/.retirement_model_cache && \
    chown -R planner:planner /home/planner/
VOLUME ["/home/planner/.retirement_model_cache"]

USER planner

# Default: run with sensible defaults
# Override at runtime: docker run retirement-planner --withdraw 8000 --sims 100000
ENTRYPOINT ["python", "retirement_model.py"]
CMD ["--help"]
