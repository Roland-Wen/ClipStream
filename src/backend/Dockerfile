# STAGE 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build tools needed for 'tokenizers'
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtualenv to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STAGE 2: Runtime
FROM python:3.11-slim as runtime

# 1. Create a non-root user for security
RUN groupadd -r clipgroup && useradd -r -g clipgroup clipuser

WORKDIR /app

# 2. Set Production Environment Variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEBUG_MODE=False \
    ENABLE_JSON_LOGS=True

# Copy the virtualenv from the builder
COPY --from=builder /opt/venv /opt/venv

# 3. Copy assets and ensure correct ownership
COPY onnx/text_model_optimized.onnx ./onnx/
COPY tokenizer.json .
COPY src/ ./src/ 

# Change ownership of the app directory to the non-root user
RUN chown -R clipuser:clipgroup /app

# 4. Switch to non-privileged user
USER clipuser

EXPOSE 8000

# 5. Production-ready command
# Using gunicorn with a single worker to avoid OOM on Render's 512MB tier
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "src.backend.main:app", "--bind", "0.0.0.0:8000"]