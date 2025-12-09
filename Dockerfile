FROM python:3.14-slim as builder


# Install build dependencies
RUN apt-get update \
&& apt-get install -y --no-install-recommends build-essential gcc git curl \
&& rm -rf /var/lib/apt/lists/*


# Create app directory
WORKDIR /app


# Copy dependency specification first for layer caching
# If you use poetry/pyproject, copy those files and adjust accordingly.
COPY requirements.txt ./


# Create virtualenv to isolate and speed up installs
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


# Install python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt


########################
# Stage 2: runtime image
########################
FROM python:3.14-slim


# Create non-root user
RUN useradd --create-home appuser
WORKDIR /app


# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


# Copy application code
COPY . /app


# Expose port (matches docker-compose mapping 8000)
EXPOSE 8000


# Recommended environment variables (can be overridden by compose or env-file)
ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1


# Use Gunicorn with Uvicorn workers for production. Adjust module path `app.main:app` to your entry point.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] # using pure uvicorn instead of Gunicorn


# Notes:
# - Ensure your project has a requirements.txt file with FastAPI, uvicorn, gunicorn, asyncpg (if using Postgres), and qdrant-client (or relevant vector client).
# - If you use poetry/pyproject.toml, modify the build stage to install poetry and run `poetry export -f requirements.txt --output requirements.txt` before installing.
# - For local development you can override CMD in docker-compose to run: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]