import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Get the folder where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Dataset path inside the same folder
data_path = os.path.join(base_dir, "att_faces")

# Load model
model = load_model(os.path.join(base_dir, "att_face_model.h5"))

# Create label mapping
label_map = {}
label_id = 0

for folder in sorted(os.listdir(data_path)):
    if os.path.isdir(os.path.join(data_path, folder)):
        label_map[label_id] = folder
        label_id += 1

# Test image
test_image_path = os.path.join(data_path, "s1", "8.pgm")

img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (92, 112))
input_face = img.flatten().reshape(1, -1) / 255.0

prediction = model.predict(input_face)
label = np.argmax(prediction)

print("Predicted Person:", label_map[label])
