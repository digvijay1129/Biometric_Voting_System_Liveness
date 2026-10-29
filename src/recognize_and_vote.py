import os
import cv2

# ---------------- DATASET CHECK ----------------
def dataset_exists():
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        return False

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0:
            return True
    return False


# ---------------- LOAD HAAR CASCADE ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- LOAD MODEL (ONLY IF DATASET EXISTS) ----------------
model = None
if dataset_exists() and os.path.exists("models/face_model.yml"):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("models/face_model.yml")
    print("✅ Dataset found. Model loaded.")
else:
    print("⚠ Dataset empty or model missing. All voters treated as NEW.")

# ---------------- SETTINGS ----------------
THRESHOLD = 70          # STRICT threshold to avoid false duplicates
TOTAL_FRAMES = 10       # Frames for decision
REQUIRED_MATCHES = 6    # Matches required for duplicate

cap = cv2.VideoCapture(0)

# ---------------- COUNTERS ----------------
match_count = 0
frame_count = 0

text = "Detecting..."
color = (255, 255, 0)   # Yellow

print("Press 'q' to exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        # -------- FACE RECOGNITION --------
        if model is not None:
            label, confidence = model.predict(face_img)
        else:
            confidence = 999   # Force NEW voter if no model

        frame_count += 1

        # Count ONLY strong matches
        if confidence < THRESHOLD:
            match_count += 1

        # -------- FINAL DECISION --------
        if frame_count >= TOTAL_FRAMES:

            if match_count >= REQUIRED_MATCHES:
                text = f"DUPLICATE VOTER (ID: {label})"
                color = (0, 0, 255)   # Red
            else:
                text = "NEW VOTER - VOTE ALLOWED"
                color = (0, 255, 0)   # Green

            # Reset counters
            frame_count = 0
            match_count = 0

        # -------- DISPLAY --------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    cv2.imshow("Phase-I Voting System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
