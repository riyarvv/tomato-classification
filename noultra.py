import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# =============================
# Load model
# =============================
interpreter = tflite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 640
CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.45

# Change this according to your ripe class index
RIPE_CLASS_ID = 1   # try 0,1,2 if needed

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    h, w, _ = frame.shape

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predictions = output[0].T   # (8400,7)

    boxes = []
    scores = []

    for pred in predictions:
        x, y, bw, bh = pred[0:4]

        class_scores = pred[4:]   # 3 classes
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]  # NO objectness multiply

        if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

            xmin = int((x - bw/2) * w / IMG_SIZE)
            ymin = int((y - bh/2) * h / IMG_SIZE)
            xmax = int((x + bw/2) * w / IMG_SIZE)
            ymax = int((y + bh/2) * h / IMG_SIZE)

            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            scores.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, scores,
                               CONF_THRESHOLD,
                               NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = scores[i]

            cv2.rectangle(original,
                          (x, y),
                          (x + bw, y + bh),
                          (0, 255, 0), 2)

            cv2.putText(original,
                        f"RIPE {conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0), 2)

    cv2.imshow("Detection", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
