"""
Service tầng nghiệp vụ cho nhận diện cảm xúc khuôn mặt.

Đóng gói lại đúng pipeline đang dùng trong real_time_detection/detect_emotion.py:
  1. Phát hiện khuôn mặt bằng Haar Cascade (OpenCV)
  2. Tiền xử lý từng khuôn mặt bằng utils.preprocess.preprocess_face
     (median blur -> CLAHE trên kênh Y (YCrCb) -> resize 224x224 -> chuẩn hóa [0,1])
  3. Suy luận bằng model Keras đã huấn luyện

Module này KHÔNG phụ thuộc FastAPI để có thể tái sử dụng cho cả REST và WebSocket,
cũng như dễ viết unit test độc lập.
"""

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.preprocess import preprocess_face

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

# Thứ tự ưu tiên tìm model khi khởi động (có thể override bằng biến môi trường EMOTION_MODEL_PATH)
DEFAULT_MODEL_CANDIDATES = [
    os.path.join(ROOT_DIR, "models", "best_model.keras"),
    os.path.join(ROOT_DIR, "models", "model_backup", "emotion_model.keras"),
    os.path.join(ROOT_DIR, "models", "best_model_cnn_optimized.keras"),
]


class ModelNotFoundError(RuntimeError):
    pass


class EmotionService:
    """
    Singleton-style service: load model + Haar cascade một lần, dùng lại cho mọi request.
    Thread-safe hoá việc load model (double-checked locking) vì FastAPI/uvicorn có thể
    xử lý nhiều request đồng thời trên nhiều worker thread.
    """

    def __init__(self, model_path: Optional[str] = None, min_face_size: int = 60):
        self._model_path_override = model_path or os.environ.get("EMOTION_MODEL_PATH")
        self._model = None
        self._model_path = None
        self._lock = threading.Lock()
        self._min_face_size = min_face_size

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError(f"Không tải được Haar cascade tại: {cascade_path}")

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _resolve_model_path(self) -> str:
        candidates = (
            [self._model_path_override] if self._model_path_override else []
        ) + DEFAULT_MODEL_CANDIDATES
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise ModelNotFoundError(
            "Không tìm thấy file model .keras nào trong: " + ", ".join(candidates)
        )

    def load(self):
        """Load model nếu chưa có. Gọi ở startup (warm-up) để tránh cold-start lúc request đầu."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                path = self._resolve_model_path()
                self._model = load_model(path)
                self._model_path = path
                # Warm-up: chạy 1 lần forward để JIT/khởi tạo graph, tránh request đầu bị chậm
                dummy = np.zeros((1, 224, 224, 3), dtype="float32")
                self._model.predict(dummy, verbose=0)
        return self._model

    @property
    def model_path(self):
        return self._model_path

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def detect_faces(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self._min_face_size, self._min_face_size),
        )
        return faces

    def predict_frame(self, frame_bgr: np.ndarray, threshold: float = 0.0):
        """
        Nhận 1 frame BGR (định dạng OpenCV mặc định), trả về danh sách khuôn mặt kèm cảm xúc.

        Trả về:
            list[dict]: mỗi phần tử có dạng
                {
                    "box": [x, y, w, h],
                    "emotion": "Happiness",
                    "confidence": 0.87,
                    "probabilities": {"Angry": 0.01, ...}
                }
        """
        model = self.load()
        t0 = time.time()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(frame_bgr)

        results = []
        for (x, y, w, h) in faces:
            face_rgb = rgb[y : y + h, x : x + w]
            if face_rgb.size == 0:
                continue

            _, _, _, model_input = preprocess_face(face_rgb)
            pred = model.predict(np.expand_dims(model_input, 0), verbose=0)[0]
            idx = int(np.argmax(pred))
            confidence = float(pred[idx])

            if confidence < threshold:
                continue

            results.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotion": EMOTION_LABELS[idx],
                    "confidence": confidence,
                    "probabilities": {
                        label: float(p) for label, p in zip(EMOTION_LABELS, pred)
                    },
                }
            )

        return {
            "faces": results,
            "count": len(results),
            "inference_ms": round((time.time() - t0) * 1000, 2),
        }


# Instance dùng chung toàn app (singleton đơn giản, không cần DI framework)
emotion_service = EmotionService()
