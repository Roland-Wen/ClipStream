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

WORKDIR /app

# Copy the virtualenv from the builder
COPY --from=builder /opt/venv /opt/venv

# Set environment paths
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy only the runtime assets
COPY onnx/text_model_optimized.onnx ./onnx/
COPY tokenizer.json .
COPY *.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]