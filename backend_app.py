"""
Backend production cho service nhận diện cảm xúc khuôn mặt thời gian thực.

Endpoints:
    GET  /health              - health check, có load model chưa, model path nào
    WS   /ws/emotion           - nhận luồng frame webcam (base64 JPEG) theo thời gian thực,
                                  trả về JSON: box, emotion, confidence cho từng khuôn mặt
    GET  /                      - trang demo dùng webcam trình duyệt qua WebSocket ở trên

Chạy local:
    pip install -r requirements.txt
    uvicorn backend_app:app --host 0.0.0.0 --port 8000

Deploy production (khuyến nghị):
    uvicorn backend_app:app --host 0.0.0.0 --port 8000 --workers 1
    (Model TensorFlow không thread-safe tốt giữa nhiều worker process khác nhau khi
    dùng chung GPU/bộ nhớ, nên nếu cần scale ngang hãy chạy nhiều container/process
    độc lập phía sau reverse proxy, thay vì --workers > 1 trong 1 container.)
"""

import base64
import binascii
import logging
import os

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from services.emotion_service import ModelNotFoundError, emotion_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-backend")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT, "static")

app = FastAPI(title="Realtime Emotion Recognition Backend", version="1.0.0")

health_app = FastAPI()

@health_app.get("/")
def health():
    return {
        "status": "ok" if emotion_service.is_ready else "model_not_loaded",
        "model_path": emotion_service.model_path,
    }

@health_app.head("/")
def health_head():
    return {
        "status": "ok" if emotion_service.is_ready else "model_not_loaded",
        "model_path": emotion_service.model_path,
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://k.mio.io.vn"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chỉ cần dùng để phục vụ demo HTML client, không bắt buộc khi tự viết client riêng
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/health", health_app)


@app.on_event("startup")
def warmup_model():
    """Preload + warm-up model khi service khởi động, tránh request/frame đầu tiên bị chậm."""
    try:
        emotion_service.load()
        logger.info("Model loaded: %s", emotion_service.model_path)
    except ModelNotFoundError as exc:
        # Không chặn service khởi động (để /health vẫn phản hồi và báo lỗi rõ ràng),
        # nhưng log rõ để dev biết cần đặt EMOTION_MODEL_PATH hoặc thêm model vào models/
        logger.error("Model chưa sẵn sàng lúc khởi động: %s", exc)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _decode_base64_frame(data_url: str) -> np.ndarray:
    """
    Nhận chuỗi base64 (có thể có prefix 'data:image/jpeg;base64,') và trả về ảnh BGR (OpenCV).
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Frame base64 không hợp lệ") from exc

    nparr = np.frombuffer(raw, dtype=np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Không giải mã được frame ảnh")
    return frame_bgr


@app.websocket("/ws/emotion")
async def ws_emotion(websocket: WebSocket):
    """
    Giao thức real-time đơn giản:

    Client (trình duyệt lấy webcam qua getUserMedia + <canvas>) gửi liên tục các
    message dạng text, mỗi message là 1 frame JPEG mã hoá base64:

        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."

    Server xử lý từng frame và trả về JSON:

        {
            "faces": [
                {"box": [x, y, w, h], "emotion": "Happiness", "confidence": 0.83,
                 "probabilities": {...}}
            ],
            "count": 1,
            "inference_ms": 42.1
        }

    Nếu frame lỗi (không decode được), server trả về {"error": "..."} và tiếp tục
    chờ frame tiếp theo (không đóng kết nối), để client tự retry mà không phải
    reconnect toàn bộ WebSocket.
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WS connected: %s", client)

    try:
        emotion_service.load()
    except ModelNotFoundError as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close(code=1011)
        return

    frame_count = 0
    try:
        while True:
            message = await websocket.receive_text()
            frame_count += 1
            try:
                frame_bgr = _decode_base64_frame(message)
                result = emotion_service.predict_frame(frame_bgr)
                await websocket.send_json(result)
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Lỗi xử lý frame #%s", frame_count)
                await websocket.send_json({"error": f"Lỗi xử lý frame: {exc}"})
    except WebSocketDisconnect:
        logger.info("WS disconnected: %s (đã xử lý %s frame)", client, frame_count)
