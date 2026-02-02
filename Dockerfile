# Dockerfile for Google Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY server/ ./server/

# Cloud Run uses PORT env variable (default 8080)
ENV PORT=8080

# Load all documents in production
ENV MAX_STARTUP_DOCS=0

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port $PORT"]
