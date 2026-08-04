import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import requests
import os
import sys

# thiết lập utils
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.preprocess import preprocess_face


def choose_model():
    Tk().withdraw()
    return filedialog.askopenfilename(
        title="Chọn mô hình",
        filetypes=[("Model", "*.keras *.h5")]
    )


def send_http_command(camera_ip, cmd, par):
    url = f"http://{camera_ip}/?custom=1&cmd={cmd}&par={par}"
    try:
        return requests.get(url, timeout=3).status_code == 200
    except:
        return False


def show_logs(logs, win_name="Logs"):
    h = max(200, len(logs) * 30)
    img = np.ones((h, 500, 3), np.uint8) * 40
    for i, t in enumerate(logs):
        cv2.putText(img, t, (10, 30 + 30*i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2)
    cv2.imshow(win_name, img)


# ==========================
# cấu hình
# ==========================

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
threshold = 0.6

camera_ip = "192.168.1.254"
rtsp_url = f"rtsp://{camera_ip}:554/xxx.mp4"

# thư mục lưu ảnh
os.makedirs("saved_faces", exist_ok=True)
for lb in emotion_labels:
    os.makedirs(f"saved_faces/{lb}", exist_ok=True)


# ==========================
# load model
# ==========================

model_path = choose_model()
if not model_path:
    print("Không chọn model.")
    exit()

model = load_model(model_path)
print("Loaded model:", model_path)

# bật camera
if not send_http_command(camera_ip, 2001, 1):
    print("Không bật được camera")
    exit()

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 16)

if not cap.isOpened():
    print("Không mở được RTSP stream")
    send_http_command(camera_ip, 2001, 0)
    exit()

logs = []


# ==========================
# bố trí cửa sổ
# ==========================

WINDOW_POS = {
    "Main": (50, 50),
    "Logs": (900, 50),
    "Median": (50, 500),
    "CLAHE": (350, 500),
    "Canny": (650, 500)
}

for name, pos in WINDOW_POS.items():
    cv2.namedWindow(name)
    cv2.moveWindow(name, pos[0], pos[1])


# ==========================
# vòng lặp chính
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_rgb = rgb[y:y+h, x:x+w]

        # pipeline
        median, clahe_rgb, edges, model_input = preprocess_face(face_rgb)

        model_input = np.expand_dims(model_input, axis=0)
        pred = model.predict(model_input, verbose=0)[0]

        idx = np.argmax(pred)
        emo = emotion_labels[idx]
        prob = pred[idx]

        # lưu ảnh nếu đạt threshold
        if prob > threshold:
            save_path = f"saved_faces/{emo}/{emo}_{int(prob*100)}_{np.random.randint(9999)}.jpg"
            cv2.imwrite(save_path, frame[y:y+h, x:x+w])

            logs.append(f"Saved: {emo} {int(prob*100)}%")
            logs = logs[-10:]

        # vẽ box & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emo} {prob:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # show các bước
        cv2.imshow("Median", cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
        cv2.imshow("CLAHE", cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow("Canny", edges)

    # show main và logs
    cv2.imshow("Main", frame)
    show_logs(logs, "Logs")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('r'):
        new_path = choose_model()
        if new_path:
            try:
                model = load_model(new_path)
                logs.append("Reloaded model")
                logs = logs[-10:]
            except Exception as e:
                logs.append(f"Load error: {e}")


cap.release()
cv2.destroyAllWindows()
send_http_command(camera_ip, 2001, 0)
