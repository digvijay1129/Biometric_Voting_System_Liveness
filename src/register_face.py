import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

user_id = input("Enter User ID (number): ")
save_path = f"dataset/user_{user_id}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0

print("Look at the camera. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]

        cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 1:
        break

cap.release()
cv2.destroyAllWindows()

print("Face registration completed.")
