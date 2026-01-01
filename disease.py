import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# 1️⃣ Load your trained model
# ----------------------------
model = load_model("tomatofinal.h5")  # Use your local path
print("✅ Model loaded successfully!")

# ----------------------------
# 2️⃣ Define class names
# ----------------------------
class_names = [
    "Healthy Tomato",
    "Blossom_End_Rot",
    "Anthracnose",
    "Bacterial_Spot",
    "Spotted_wilt_Virus"
]

# ----------------------------
# 3️⃣ Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default laptop camera

# Optional: set camera width & height
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (224, 224))  # match your model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction[0])
    disease_class = class_names[class_idx]

    # Binary prediction
    if disease_class.lower() == "healthy tomato":
        binary_pred = "Healthy"
        binary_value = 0
    else:
        binary_pred = "Unhealthy"
        binary_value = 1

    # Display predictions on video
    label = f"{binary_pred} ({disease_class}): {np.max(prediction)*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if binary_value==0 else (0, 0, 255), 2)
    
    cv2.imshow("Tomato Disease Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()