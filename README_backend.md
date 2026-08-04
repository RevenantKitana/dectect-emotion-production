# Backend Real-time Emotion Recognition — hướng dẫn deploy

Backend FastAPI nhận diện cảm xúc khuôn mặt, chỉ hỗ trợ:
- `WS /ws/emotion` — nhận luồng frame webcam theo thời gian thực, trả kết quả (box + cảm xúc + confidence) ngay lập tức, cùng ý tưởng xử lý với `real_time_detection/detect_emotion.py` nhưng chạy phía server để có thể deploy dùng chung cho nhiều client qua trình duyệt, thay vì mở cửa sổ `cv2.imshow` cục bộ.
- `GET /` — trang demo dùng webcam ngay trên trình duyệt (`static/index.html`) để test nhanh WebSocket ở trên.

## Kiến trúc

```
backend_app.py              FastAPI app: REST + WebSocket + serve demo HTML
services/emotion_service.py Business logic: load model 1 lần (singleton, warm-up),
                             Haar cascade detect multi-face, tiền xử lý + predict
utils/preprocess.py         Pipeline tiền xử lý gốc (median blur -> CLAHE -> resize 224x224)
static/index.html           Demo client: getUserMedia -> canvas -> base64 JPEG -> WebSocket
```

Khác với script `detect_emotion.py` gốc (chạy cục bộ, mở nhiều cửa sổ cv2.imshow,
chọn model qua dialog), backend này:
- Load model **một lần khi khởi động** (`@app.on_event("startup")`) kèm warm-up
  bằng 1 lần predict giả để tránh request/frame đầu tiên bị chậm.
- Hỗ trợ **nhiều khuôn mặt** trong 1 frame (trả về mảng `faces`), không chỉ khuôn mặt đầu tiên.
- Không phụ thuộc `cv2.imshow`/Tkinter (chạy headless trong container/server, không cần màn hình).
- Webcam được truy cập **ở phía trình duyệt của người dùng** (qua `getUserMedia`),
  không phải webcam gắn vào server — đúng mô hình triển khai web thực tế.

## Chạy local

```bash
pip install -r requirements-backend.txt
uvicorn backend_app:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: `http://localhost:8000/` → bấm "Bắt đầu" để cấp quyền webcam và xem
nhận diện cảm xúc theo thời gian thực (bounding box + nhãn cảm xúc + % confidence
vẽ đè trực tiếp lên canvas).

## Chạy bằng Docker

```bash
docker build -t emotion-backend .
docker run -p 8000:8000 emotion-backend
```

Mặc định container dùng model tại `models/best_model.keras`. Muốn đổi model, set biến
môi trường `EMOTION_MODEL_PATH`:

```bash
docker run -p 8000:8000 -e EMOTION_MODEL_PATH=/app/models/model_backup/emotion_model.keras emotion-backend
```

## API

### `GET /health`
```json
{ "status": "ok", "model_path": "/app/models/best_model.keras" }
```

### `WS /ws/emotion`
Giao thức text-frame đơn giản, không cần thư viện client đặc biệt:

- Client gửi liên tục các message text dạng:
  `data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...`
  (đúng định dạng `canvas.toDataURL('image/jpeg', quality)` trả về trong trình duyệt)
- Server trả về JSON cùng cấu trúc `{"faces": [...], "count": ..., "inference_ms": ...}`
  hoặc `{"error": "..."}` nếu frame lỗi (kết nối **không bị đóng** khi 1 frame lỗi,
  để client tiếp tục gửi frame tiếp theo mà không cần reconnect).

Client mẫu đầy đủ: xem `static/index.html`.

## Hướng dẫn tích hợp client

### 1. Kết nối WebSocket
Backend chỉ hỗ trợ giao thức WebSocket cho luồng thời gian thực.

- Local: `ws://localhost:8000/ws/emotion`
- Production với HTTPS: `wss://your-domain/ws/emotion`

Ví dụ trên trình duyệt:

```js
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws/emotion`);

ws.onopen = () => console.log('Connected');
ws.onerror = (err) => console.error('WebSocket error', err);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.error) {
    console.error(data.error);
    return;
  }

  console.log('Faces:', data.faces);
  console.log('Count:', data.count);
  console.log('Inference ms:', data.inference_ms);
};
```

### 2. Gửi frame từ camera hoặc canvas
Mỗi tin nhắn là một chuỗi base64 JPEG, đúng định dạng mà `canvas.toDataURL('image/jpeg', quality)` trả về.

```js
const canvas = document.createElement('canvas');
canvas.width = 320;
canvas.height = 240;

const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
ws.send(dataUrl);
```

### 3. Cấu trúc phản hồi
Server sẽ trả về JSON theo mẫu sau:

```json
{
  "faces": [
    {
      "box": [120, 80, 150, 150],
      "emotion": "Happiness",
      "confidence": 0.87,
      "probabilities": {
        "Angry": 0.01,
        "Neutral": 0.08
      }
    }
  ],
  "count": 1,
  "inference_ms": 38.4
}
```

Nếu có lỗi frame hoặc model chưa sẵn sàng, server sẽ trả về:

```json
{ "error": "..." }
```

### 4. Khuyến nghị khi tích hợp
- Gửi frame ở tần suất vừa phải, ví dụ 8–10 FPS, để tránh hàng đợi quá dài.
- Chỉ gửi frame mới sau khi nhận phản hồi của frame trước đó.
- Với HTTPS/WSS, hãy cấu hình reverse proxy để chuyển tiếp WebSocket đúng cách.
- Nếu dùng nhiều client đồng thời, nên chạy nhiều instance backend thay vì tăng worker trong cùng một container.

## Triển khai với Render

Render có thể host backend FastAPI này như một service web. Vì app dùng WebSocket, hãy đảm bảo service chạy đúng port và không bị chặn bởi proxy.

### 1. Tạo service trên Render
- Chọn loại service: Web Service
- Kết nối repository này
- Build Command:

```bash
pip install -r requirements-backend.txt
```

- Start Command:

```bash
uvicorn backend_app:app --host 0.0.0.0 --port $PORT
```

### 2. Biến môi trường
Đặt ít nhất:

```bash
EMOTION_MODEL_PATH=/opt/render/project/src/models/best_model.keras
```

Nếu bạn dùng model khác, đổi đường dẫn tương ứng.

### 3. Render và WebSocket
Render hỗ trợ WebSocket tốt khi backend trả về đúng URL `/ws/emotion` và lắng nghe trên biến môi trường `$PORT`.

Client phía trình duyệt nên dùng:

```js
const ws = new WebSocket(`wss://your-render-service.onrender.com/ws/emotion`);
```

### 4. Lưu ý quan trọng
- Render cần chạy service với `uvicorn` đúng như trên, không dùng `python backend_app.py`.
- Nếu dùng HTTPS/WSS thì client phải dùng `wss://`, không phải `ws://`.
- Nếu model quá nặng, hãy cân nhắc dùng instance có đủ RAM/CPU hoặc tối ưu giảm kích thước frame trước khi gửi.

## Ghi chú production

- **1 worker / container**: model TensorFlow được load 1 lần trong process; nếu cần
  tăng throughput, scale ngang bằng nhiều container phía sau load balancer/reverse
  proxy (nginx, Traefik, hoặc service mesh trên K8s), thay vì tăng `--workers` trong
  cùng 1 container.
- **Giảm FPS gửi lên server**: demo gửi ~10 FPS (mỗi 100ms) và chỉ gửi frame mới sau
  khi đã nhận kết quả frame trước, tránh dồn ứ hàng đợi khi model chậm hơn webcam.
- **Ảnh gửi lên được resize nhỏ (320x240)** trước khi encode JPEG để giảm băng thông
  và tăng tốc độ xử lý; bounding box được scale ngược lại kích thước hiển thị ở client.
- Còn thiếu cho production quy mô lớn (tuỳ nhu cầu thực tế bổ sung thêm):
  - Authentication / rate limiting cho endpoint WebSocket và REST
  - GPU inference + batch nhiều frame nếu nhiều client đồng thời
  - Structured logging, metrics (Prometheus), tracing
  - HTTPS/WSS termination (qua reverse proxy) — bắt buộc để `getUserMedia` hoạt động
    trên domain không phải localhost
