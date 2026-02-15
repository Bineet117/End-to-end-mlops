FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY __init__.py .
COPY src/ ./src/
COPY configs/ ./configs/
COPY loggings/ ./loggings/
COPY serve/ ./serve/
COPY pyproject.toml .

# Create directories
RUN mkdir -p data/raw data/processed models

# Expose port 8080 (Vertex AI default)
EXPOSE 8080

# Vertex AI sends health checks and predictions to port 8080
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
