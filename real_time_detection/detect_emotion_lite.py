import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ============================
#   CONFIG
# ============================

CAM_WIDTH = 640
CAM_HEIGHT = 480

# Layout window (không overlap)
WIN_POS = {
    "cam":   (0, 0),
    "logs":  (0, 520),
    "median": (680, 0),
    "clahe":  (680, 260),
    "edges":  (680, 520),
}

# ============================
#   INIT MODELS + UTILS
# ============================

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
emotion_model = load_model("emotion_model.h5")

EMOTION_LABELS = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

def apply_median_filter(img):
    return cv2.medianBlur(img, 5)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def compute_edges(gray):
    return cv2.Canny(gray, 80, 180)

def show_logs(lines):
    h = 350
    w = 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    y = 20
    for t in lines[-15:]:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 22

    cv2.imshow("Logs", img)


# ============================
#   MAIN
# ============================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

logs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------------
    #  PROCESS ENHANCEMENT WINDOWS
    # ----------------------------------
    median_img = apply_median_filter(frame)
    clahe_img = apply_clahe(gray)
    edges_img = compute_edges(gray)

    # ----------------------------------
    #  EMOTION DETECTION (CROPPED FACE)
    # ----------------------------------
    face_crop = None

    if results.multi_face_landmarks:
        h, w = gray.shape
        xs = []
        ys = []

        for lm in results.multi_face_landmarks[0].landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        x1, y1 = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
        x2, y2 = min(max(xs) + 20, w), min(max(ys) + 20, h)

        face_crop = gray[y1:y2, x1:x2]

        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, (48, 48))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=(0, -1))

            preds = emotion_model.predict(face_resized, verbose=0)[0]
            emotion = EMOTION_LABELS[np.argmax(preds)]
            score = np.max(preds)

            logs.append(f"{emotion} {score:.2f}")

            cv2.putText(frame, f"{emotion}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
    else:
        logs.append("No face detected")

    # ============================
    #   SHOW WINDOWS + FIX POS
    # ============================

    cv2.imshow("Emotion Detection", frame)
    cv2.moveWindow("Emotion Detection", *WIN_POS["cam"])

    cv2.imshow("Median", cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))
    cv2.moveWindow("Median", *WIN_POS["median"])

    cv2.imshow("CLAHE", clahe_img)
    cv2.moveWindow("CLAHE", *WIN_POS["clahe"])

    cv2.imshow("Edges", edges_img)
    cv2.moveWindow("Edges", *WIN_POS["edges"])

    show_logs(logs)
    cv2.moveWindow("Logs", *WIN_POS["logs"])

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
