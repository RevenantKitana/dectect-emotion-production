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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.emotion_service import ModelNotFoundError, emotion_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-backend")

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT, "static")

app = FastAPI(title="Realtime Emotion Recognition Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "3.12.251.153",
        "3.20.63.178",
        "3.77.67.4",
        "3.79.134.69",
        "3.105.133.239",
        "3.105.190.221",
        "3.133.226.214",
        "3.149.57.90",
        "3.212.128.62",
        "5.161.61.238",
        "5.161.73.160",
        "5.161.75.7",
        "5.161.113.195",
        "5.161.117.52",
        "5.161.177.47",
        "5.161.194.92",
        "5.161.215.244",
        "5.223.43.32",
        "5.223.53.147",
        "5.223.57.22",
        "18.116.205.62",
        "18.180.208.214",
        "18.192.166.72",
        "18.193.252.127",
        "24.144.78.39",
        "24.144.78.185",
        "34.198.201.66",
        "45.55.123.175",
        "45.55.127.146",
        "49.13.24.81",
        "49.13.130.29",
        "49.13.134.145",
        "49.13.164.148",
        "49.13.167.123",
        "52.15.147.27",
        "52.22.236.30",
        "52.28.162.93",
        "52.59.43.236",
        "52.87.72.16",
        "54.64.67.106",
        "54.79.28.129",
        "54.87.112.51",
        "54.167.223.174",
        "54.249.170.27",
        "63.178.84.147",
        "64.225.81.248",
        "64.225.82.147",
        "78.46.190.63",
        "78.46.215.1",
        "78.47.98.55",
        "78.47.173.76",
        "88.99.80.227",
        "91.99.101.207",
        "128.140.41.193",
        "128.140.106.114",
        "129.212.132.140",
        "134.199.240.137",
        "138.197.53.117",
        "138.197.53.138",
        "138.197.54.143",
        "138.197.54.247",
        "138.197.63.92",
        "139.59.50.44",
        "142.132.180.39",
        "143.198.249.237",
        "143.198.250.89",
        "143.244.196.21",
        "143.244.196.211",
        "143.244.221.177",
        "144.126.251.21",
        "146.190.9.187",
        "152.42.149.135",
        "157.90.155.240",
        "157.90.156.63",
        "159.69.158.189",
        "159.223.243.219",
        "161.35.247.201",
        "167.99.18.52",
        "167.235.143.113",
        "168.119.53.160",
        "168.119.96.239",
        "168.119.123.75",
        "170.64.250.64",
        "170.64.250.132",
        "170.64.250.235",
        "178.156.181.172",
        "178.156.184.20",
        "178.156.185.127",
        "178.156.185.231",
        "178.156.187.238",
        "178.156.189.113",
        "178.156.189.249",
        "188.166.201.79",
        "206.189.241.133",
        "209.38.49.1",
        "209.38.49.206",
        "209.38.49.226",
        "209.38.51.43",
        "209.38.53.7",
        "209.38.124.252",
        "216.144.248.27",
        "216.144.248.28",
        "216.144.248.29",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chỉ cần dùng để phục vụ demo HTML client, không bắt buộc khi tự viết client riêng
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/health")
def health():
    return {
        "status": "ok" if emotion_service.is_ready else "model_not_loaded",
        "model_path": emotion_service.model_path,
    }


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
