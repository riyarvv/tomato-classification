import cv2
import numpy as np

# =============================
# Load TFLite model
# =============================
from tflite_runtime.interpreter import Interpreter
print("✅ Using tflite_runtime (Raspberry Pi 5)")

# =============================
# Initialize TFLite interpreter
# =============================
MODEL_PATH = "tomato_model_pi.tflite"  
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded")

HEALTHY_CLASS_INDEX = 0  # "Healthy Tomato"

# =============================
# Open webcam
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Ripe + Healthy/Unhealthy Detection Started (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =============================
    # Ripe detection (HSV)
    # =============================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # ignore small contours
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        tomato_crop = frame[y:y+h, x:x+w]

        if tomato_crop.size == 0:
            continue

        # =============================
        # Preprocess for TFLite
        # =============================
        img = cv2.resize(tomato_crop, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Get output
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx] * 100

        # =============================
        # Healthy vs Unhealthy logic
        # =============================
        if class_idx == HEALTHY_CLASS_INDEX and confidence >= 60:
            label = "Healthy"
            color = (0, 255, 0)
        elif class_idx != HEALTHY_CLASS_INDEX and confidence >= 70:
            label = "Unhealthy"
            color = (0, 0, 255)
        else:
            label = "Healthy"
            color = (0, 255, 0)

        # =============================
        # Draw results
        # =============================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, "Ripe", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Ripe Tomato + Health Status (TFLite)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

