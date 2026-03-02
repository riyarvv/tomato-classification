import cv2
import numpy as np
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
# Start Camera
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected ❌")
    exit()

print("Camera Started ✅")
print("Detecting ONLY RIPE 🍅")
print("Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Resize to model input
    img = cv2.resize(frame, (input_w, input_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # [7,8400]
    output = output.T  # [8400,7]

    for pred in output:
        x, y, bw, bh, score, class_id, _ = pred
        class_id = int(round(class_id))

        if score > 0.30 and class_id == 2:  # Only ripe
            xmin = int((x - bw/2) * w)
            ymin = int((y - bh/2) * h)
            xmax = int((x + bw/2) * w)
            ymax = int((y + bh/2) * h)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(frame, f"Ripe {score:.2f}", (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Ripe Detection 🍅", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
