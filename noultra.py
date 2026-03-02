import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_float16.tflite"   # change to your model
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640

# ==============================
# LOAD MODEL
# ==============================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==============================
# CAMERA SETUP
# ==============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 960, 720)

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# NMS FUNCTION
# ==============================
def non_max_suppression(boxes, scores):
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        IOU_THRESHOLD
    )
    return indices

# ==============================
# MAIN LOOP
# ==============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_h, original_w = frame.shape[:2]

    # Preprocess
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = []
    scores = []
    class_ids = []

    for det in output:
        confidence = det[4]
        if confidence < CONF_THRESHOLD:
            continue

        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score < CONF_THRESHOLD:
            continue

        # YOLO format: x_center, y_center, w, h
        x_center, y_center, w, h = det[0:4]

        x = int((x_center - w / 2) * original_w)
        y = int((y_center - h / 2) * original_h)
        width = int(w * original_w)
        height = int(h * original_h)

        boxes.append([x, y, width, height])
        scores.append(float(score))
        class_ids.append(class_id)

    indices = non_max_suppression(boxes, scores)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Class {class_ids[i]}: {scores[i]:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
