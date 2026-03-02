import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2  # Confirmed from your working ultralytics test

# ==========================================
# Load Model
# ==========================================
print("Loading YOLO TFLite model...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

print("Model Loaded ✅")

# ==========================================
# Open USB Camera (Stable for Pi)
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not detected ❌")
    exit()

print("Ripeness Detection Started 🍅 (Press Q to Quit)")

prev_time = 0

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w, _ = frame.shape

    # --------------------------------------
    # Preprocess
    # --------------------------------------
    img = cv2.resize(frame, (input_w, input_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # --------------------------------------
    # Inference
    # --------------------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.T  # shape: [8400, 7]

    boxes = []
    scores = []

    for pred in output:
        x, y, w, h, conf, cls_id, _ = pred
        cls_id = int(cls_id)

        if conf > CONF_THRESHOLD and cls_id == RIPE_CLASS_ID:
            xmin = int((x - w / 2) * orig_w)
            ymin = int((y - h / 2) * orig_h)
            xmax = int((x + w / 2) * orig_w)
            ymax = int((y + h / 2) * orig_h)

            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            scores.append(float(conf))

    # --------------------------------------
    # Non-Max Suppression (Very Important)
    # --------------------------------------
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            score = scores[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Ripe {score:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # --------------------------------------
    # FPS Counter
    # --------------------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    # Bigger display window
    display = cv2.resize(frame, (960, 720))
    cv2.imshow("YOLOv8 Ripeness Detection 🍅", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
