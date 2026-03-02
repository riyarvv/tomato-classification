import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ==============================
# Load TFLite Model
# ==============================
interpreter = tflite.Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Loaded ✅")

# ==============================
# Settings
# ==============================
IMG_SIZE = 640
CONF_THRESHOLD = 0.4
RIPE_CLASS_ID = 2   # change if needed

# ==============================
# Start Camera
# ==============================
cap = cv2.VideoCapture(0)

print("Camera Started")
print("Detecting ONLY RIPE 🍅")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()
    h, w, _ = frame.shape

    # Resize
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predictions = output[0]   # shape (7, 8400)

    for i in range(predictions.shape[1]):

        x = predictions[0][i]
        y = predictions[1][i]
        w_box = predictions[2][i]
        h_box = predictions[3][i]
        obj_conf = predictions[4][i]

        class_scores = predictions[5:, i]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]

        confidence = obj_conf * class_conf

        if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

            # Convert from center format to box format
            xmin = int((x - w_box / 2) * w / IMG_SIZE)
            ymin = int((y - h_box / 2) * h / IMG_SIZE)
            xmax = int((x + w_box / 2) * w / IMG_SIZE)
            ymax = int((y + h_box / 2) * h / IMG_SIZE)

            # Draw box
            cv2.rectangle(original, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(original, f"RIPE {confidence:.2f}",
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    cv2.imshow("Ripe Detection", original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
