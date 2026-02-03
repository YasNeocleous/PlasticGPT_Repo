# Dockerfile for Google Cloud Run
# Multi-stage build: Frontend + Backend

# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/client
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
# Skip tsc type-check (paths don't resolve in Docker), just run vite build
RUN npx vite build

# Stage 2: Python backend with built frontend
FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY server/ ./server/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/client/dist ./static

# Cloud Run uses PORT env variable (default 8080)
ENV PORT=8080

# Load all documents in production
ENV MAX_STARTUP_DOCS=0

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port $PORT"]
