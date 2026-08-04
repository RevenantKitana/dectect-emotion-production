import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
import requests
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.preprocess import preprocess_face


# =============================
# SYSTEM CONFIG
# =============================
camera_ip = "192.168.1.254"
rtsp_url = f"rtsp://{camera_ip}:554/xxx.mp4"
emotion_labels = ['Angry','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def send_http_command(cmd, par):
    try:
        url = f"http://{camera_ip}/?custom=1&cmd={cmd}&par={par}"
        return requests.get(url, timeout=2).status_code == 200
    except:
        return False

# turn on camera
send_http_command(2001, 1)

# load model
model = load_model("model.keras")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)

if not cap.isOpened():
    print("Không mở được camera.")
    exit()

# =============================
# TRACKBAR CONTROL WINDOW
# =============================
cv2.namedWindow("Ctrl")
cv2.resizeWindow("Ctrl", 400, 200)

cv2.createTrackbar("Threshold", "Ctrl", 60, 100, lambda x: None)
cv2.createTrackbar("Scale x100", "Ctrl", 110, 200, lambda x: None)
cv2.createTrackbar("Neighbors", "Ctrl", 5, 20, lambda x: None)
cv2.createTrackbar("Save", "Ctrl", 0, 1, lambda x: None)
cv2.createTrackbar("Mode", "Ctrl", 0, 3, lambda x: None)
cv2.createTrackbar("Canny", "Ctrl", 80, 255, lambda x: None)

debug = False

# =============================
# LOOP
# =============================
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # lấy tham số
    threshold = cv2.getTrackbarPos("Threshold", "Ctrl") / 100
    scale = cv2.getTrackbarPos("Scale x100", "Ctrl") / 100
    neighbors = cv2.getTrackbarPos("Neighbors", "Ctrl")
    save_enable = cv2.getTrackbarPos("Save", "Ctrl")
    mode = cv2.getTrackbarPos("Mode", "Ctrl")
    canny_th = cv2.getTrackbarPos("Canny", "Ctrl")

    # detect face (ảnh grayscale nhẹ nhất)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale, neighbors)

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

        median, clahe_rgb, edges, model_input = preprocess_face(
            face_rgb, 
            mode=mode, 
            canny_th=canny_th
        )

        model_input = np.expand_dims(model_input, axis=0)

        pred = model.predict(model_input, verbose=0)[0]
        idx = np.argmax(pred)
        emo = emotion_labels[idx]
        prob = pred[idx]

        if prob > threshold:
            color = (0,255,0)
            if save_enable:
                os.makedirs(f"saved_faces/{emo}", exist_ok=True)
                cv2.imwrite(
                    f"saved_faces/{emo}/{emo}_{int(prob*100)}.jpg",
                    frame[y:y+h, x:x+w]
                )
        else:
            color = (120,120,120)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{emo} {prob:.2f}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # DEBUG windows chỉ bật khi cần
        if debug:
            cv2.imshow("Median", cv2.cvtColor(median, cv2.COLOR_RGB2BGR))
            cv2.imshow("CLAHE", cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR))
            cv2.imshow("Edges", edges)

    # tính FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"FPS {fps:.1f}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Main", frame)
    cv2.imshow("Ctrl", np.zeros((1,1), np.uint8))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('d'):
        debug = not debug
        if not debug:
            cv2.destroyWindow("Median")
            cv2.destroyWindow("CLAHE")
            cv2.destroyWindow("Edges")

cap.release()
cv2.destroyAllWindows()
send_http_command(2001, 0)
