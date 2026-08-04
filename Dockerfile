FROM python:3.10-slim

# Thư viện hệ thống cần cho OpenCV (headless) chạy được trong container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    EMOTION_MODEL_PATH=/app/models/best_model.keras

EXPOSE 8000

# 1 worker: TensorFlow model được load 1 lần / process, tránh nhân bản bộ nhớ nhiều lần.
# Muốn scale ngang -> chạy nhiều container phía sau load balancer / reverse proxy.
CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
