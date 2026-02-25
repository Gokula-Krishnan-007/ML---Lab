import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data_path = r"C:\Users\Gokul\Desktop\VI\Machine Learning\Lab\Experiments\ML---Lab\Lab programs\Face Recognition\att_faces"



faces = []
labels = []
label_map = {}

label_id = 0

# Load dataset
for folder in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder)
    
    if os.path.isdir(folder_path):
        label_map[label_id] = folder
        
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".pgm"):
                img_path = os.path.join(folder_path, file)
                
                # Read PGM image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize (optional, but keeping original size is fine)
                img = cv2.resize(img, (92, 112))
                
                # Flatten image
                faces.append(img.flatten())
                labels.append(label_id)
        
        label_id += 1

faces = np.array(faces) / 255.0
labels = np.array(labels)

# Convert labels to categorical
labels = to_categorical(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    faces, labels, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Build ANN Model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(10304,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(40, activation='softmax'))  # 40 persons

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train,
          epochs=50,
          batch_size=16,
          validation_data=(X_test, y_test))

# Save model
model.save("att_face_model.h5")

print("Model training completed and saved!")
