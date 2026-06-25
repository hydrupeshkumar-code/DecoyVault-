FROM python:3.11-slim

WORKDIR /app

# System dependencies for rasterio / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY ai/requirements.txt /app/ai/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r ai/requirements.txt -r backend/requirements.txt

# Application code
COPY ai/ /app/ai/
COPY backend/ /app/backend/

ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
