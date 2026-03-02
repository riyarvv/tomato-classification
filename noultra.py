import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2  # confirmed from ultralytics test

# Load model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("YOLO Ripeness Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Ripeness Detection", 960, 720)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocess
    img = cv2.resize(frame, (input_w, input_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # (7,8400)
    output = output.T  # -> (8400,7)

    print("Max confidence:", np.max(output[:,4]))
    print("Unique class IDs:", np.unique(output[:,5]))

    boxes = []
    scores = []

    for pred in output:
        x, y, w, h, conf, class_id, _ = pred
        class_id = int(class_id)

        if conf > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:
            xmin = int((x - w/2) * orig_w)
            ymin = int((y - h/2) * orig_h)
            xmax = int((x + w/2) * orig_w)
            ymax = int((y + h/2) * orig_h)

            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            scores.append(float(conf))

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            score = scores[i]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Ripe {score:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    cv2.imshow("YOLO Ripeness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
