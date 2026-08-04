import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess import preprocess_face

# Model picker
def choose_model():
    Tk().withdraw()
    return filedialog.askopenfilename(
        title="Chọn file mô hình",
        filetypes=[("Keras model", "*.keras *.h5")]
    )

# Log window
def show_logs(logs):
    h = max(200, len(logs) * 30)
    w = 500
    frame = np.ones((h, w, 3), dtype=np.uint8) * 50
    for i, text in enumerate(logs):
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Logs", frame)


# ================================================
#                     MAIN
# ================================================

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

threshold = 0.6
save_dir = "saved_faces"

os.makedirs(save_dir, exist_ok=True)
for label in emotion_labels:
    os.makedirs(os.path.join(save_dir, label), exist_ok=True)

model_path = choose_model()
if not model_path:
    print("Không chọn model.")
    exit()

model = load_model(model_path)

cap = cv2.VideoCapture(0)
logs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_rgb = rgb[y:y+h, x:x+w]

        # Preprocessing
        median_img, clahe_img, edges_img, model_input = preprocess_face(face_rgb)

        # Prediction
        pred = model.predict(np.expand_dims(model_input, 0), verbose=0)[0]
        idx = np.argmax(pred)
        emotion = emotion_labels[idx]

        # Windows
        cv2.imshow("Median", cv2.cvtColor(median_img, cv2.COLOR_RGB2BGR))
        cv2.moveWindow("Median", 750, 0)

        cv2.imshow("CLAHE", cv2.cvtColor(clahe_img, cv2.COLOR_RGB2BGR))
        cv2.moveWindow("CLAHE", 750, 280)

        cv2.imshow("Edges", edges_img)
        cv2.moveWindow("Edges", 750, 560)

        cv2.imshow("Emotion Detection", frame)
        cv2.moveWindow("Emotion Detection", 0, 0)

        show_logs(logs)
        cv2.moveWindow("Logs", 0, 500)

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # =========================================
        #        SAVE 4 VERSIONS (ADDED)
        # =========================================
        if pred[idx] > threshold:
            save_path = os.path.join(save_dir, emotion)

            base_name = f"{emotion}_{int(pred[idx] * 100)}_{np.random.randint(9999)}"

            raw_path = os.path.join(save_path, base_name + "_raw.jpg")
            median_path = os.path.join(save_path, base_name + "_median.jpg")
            clahe_path = os.path.join(save_path, base_name + "_clahe.jpg")
            edges_path = os.path.join(save_path, base_name + "_edges.jpg")

            # SAVE RAW
            cv2.imwrite(raw_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
            # SAVE MEDIAN
            cv2.imwrite(median_path, cv2.cvtColor(median_img, cv2.COLOR_RGB2BGR))
            # SAVE CLAHE
            cv2.imwrite(clahe_path, cv2.cvtColor(clahe_img, cv2.COLOR_RGB2BGR))
            # SAVE EDGES
            cv2.imwrite(edges_path, edges_img)

            logs.append(f"Saved: {base_name} (4 versions)")
            logs = logs[-10:]

    # Key event
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
