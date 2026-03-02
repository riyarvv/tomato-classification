import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "best_float16.tflite"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
RIPE_CLASS_ID = 1   # Change if your ripe class index is different

print("Loading TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Loaded ✅")

# ==============================
# SAFE CAMERA OPEN
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Camera failed to open")
    exit()

print("Camera Started ✅")
print("Press Q to Quit")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed")
        break

    original = frame.copy()
    h, w, _ = frame.shape

    # ------------------------------
    # PREPROCESS
    # ------------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # ------------------------------
    # INFERENCE
    # ------------------------------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Expected shape: (1, 7, 8400)
    predictions = output[0].T  # shape becomes (8400, 7)

    boxes = []
    scores = []

    # ------------------------------
    # DECODE OUTPUT
    # ------------------------------
    for pred in predictions:
        x, y, bw, bh, obj_conf, class0, class1 = pred

        class_scores = np.array([class0, class1])
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        confidence = obj_conf * class_conf

        if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

            xmin = (x - bw / 2) * w / IMG_SIZE
            ymin = (y - bh / 2) * h / IMG_SIZE
            xmax = (x + bw / 2) * w / IMG_SIZE
            ymax = (y + bh / 2) * h / IMG_SIZE

            boxes.append([
                int(xmin),
                int(ymin),
                int(xmax - xmin),
                int(ymax - ymin)
            ])
            scores.append(float(confidence))

    # ------------------------------
    # APPLY NMS
    # ------------------------------
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = scores[i]

            cv2.rectangle(
                original,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                original,
                f"RIPE {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Ripe Detection", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEAN EXIT
# ==============================
cap.release()
cv2.destroyAllWindows()
print("Camera Closed ✅")
