import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []

for user_folder in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user_folder)
    label = int(user_folder.split("_")[1])

    for image_name in os.listdir(user_path):
        img_path = os.path.join(user_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces.append(img)
        labels.append(label)

# IMPORTANT FIX: labels must be numpy array
labels = np.array(labels, dtype=np.int32)

# Create LBPH recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Save trained model
model.save("models/face_model.yml")

print("Model training completed and saved successfully.")
