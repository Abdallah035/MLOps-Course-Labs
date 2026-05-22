# Bank Customer Churn Prediction API — container image
# MLOps Course Lab 2 — Abdallah Mohamed
#
# Uses uv for fast, reproducible installs (respects uv.lock).

FROM python:3.12-slim

# Avoid .pyc files and force unbuffered stdout (so logs appear live)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv (the dependency manager used in Lab 1)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (layer caching): only re-runs when deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the application code and model artifact
COPY main.py ./
COPY app/ ./app/
COPY data/ ./data/

# Litestar will serve on this port inside the container
EXPOSE 8000

# Run the API with uvicorn via litestar, binding to all interfaces
CMD ["uv", "run", "litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]
