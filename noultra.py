import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# ==========================
# Load TFLite Model
# ==========================
interpreter = tflite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

print("Model Loaded ✅")

# ==========================
# Start USB Camera (V4L2 for Pi stability)
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Force resolution (best balance for Pi)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not detected ❌")
    exit()

print("Camera Started ✅")
print("Detecting ONLY RIPE 🍅")
print("Press Q to Quit")

prev_time = 0

# ==========================
# Main Loop
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w, _ = frame.shape

    # Resize for model
    img = cv2.resize(frame, (input_w, input_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.T  # shape [8400, 7]

    # Faster filtering (vectorized)
    scores = output[:, 4]
    class_ids = output[:, 5].astype(int)

    # Only ripe (class 2) and confidence > 0.30
    mask = (scores > 0.30) & (class_ids == 2)
    detections = output[mask]

    for det in detections:
        x, y, bw, bh, score, class_id, _ = det

        xmin = int((x - bw/2) * orig_w)
        ymin = int((y - bh/2) * orig_h)
        xmax = int((x + bw/2) * orig_w)
        ymax = int((y + bh/2) * orig_h)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        cv2.putText(frame, f"Ripe {score:.2f}", (xmin, ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ==========================
    # FPS Calculation
    # ==========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Bigger display window
    display = cv2.resize(frame, (960, 720))
    cv2.imshow("Ripe Detection 🍅", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
