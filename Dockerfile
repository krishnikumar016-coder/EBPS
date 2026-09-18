# Use official slim Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build-essential, libpq-dev for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn psycopg2-binary gunicorn

# Copy application code
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Default command to run FastAPI app with Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
