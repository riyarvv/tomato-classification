import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1️⃣ ARM INITIALIZATION
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH = 0, 1, 2, 3, 5

servos = {}
for ch in [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH]:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

LIMITS = {
    BASE_CH:     {"neutral": 20,  "pick": 40},
    PITCH_CH:    {"neutral": 90},
    SHOULDER_CH: {"neutral": 125, "pick": 115},
    ELBOW_CH:    {"neutral": 30,  "pick": 50},
    GRIPPER_CH:  {"open": 170,    "close": 20}
}

def move_slow(channel_id, target_angle, speed=0.02):
    current = servos[channel_id].angle
    if current is None:
        current = 90

    start = int(current)
    target = int(target_angle)

    step = 1 if target > start else -1

    for angle in range(start, target + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

def go_home():
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_and_drop():
    print("🍅 Picking Ripe Tomato...")
    move_slow(BASE_CH, LIMITS[BASE_CH]["pick"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"])
    time.sleep(1)

    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    go_home()

# ==========================================
# 2️⃣ LOAD NEW YOLO TFLITE MODEL
# ==========================================
MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2   # confirmed ripe class

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# ==========================================
# 3️⃣ CAMERA
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Harvest Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Harvest Vision", 960, 720)

go_home()
prev_time = 0

# ==========================================
# 4️⃣ MAIN LOOP
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # -------- DRAW CENTER TARGET ZONE --------
        center_x, center_y = orig_w // 2, orig_h // 2
        zone_size = 120

        zone_left = center_x - zone_size // 2
        zone_right = center_x + zone_size // 2
        zone_top = center_y - zone_size // 2
        zone_bottom = center_y + zone_size // 2

        cv2.rectangle(frame, (zone_left, zone_top),
                      (zone_right, zone_bottom),
                      (255,255,255), 1)

        # -------- PREPROCESS --------
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        output = output.T

        boxes = []
        scores = []
        centers = []

        for pred in output:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]

            class_id = int(np.argmax(class_scores))
            confidence = class_scores[class_id]

            if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

                xmin = int((x - w/2) * orig_w)
                ymin = int((y - h/2) * orig_h)
                xmax = int((x + w/2) * orig_w)
                ymax = int((y + h/2) * orig_h)

                boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                scores.append(float(confidence))
                centers.append((int(x * orig_w), int(y * orig_h)))

        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                   CONF_THRESHOLD, IOU_THRESHOLD)

        triggered = False

        if len(indices) > 0:
            for idx in indices.flatten():

                x, y, bw, bh = boxes[idx]
                score = scores[idx]
                cx, cy = centers[idx]

                is_centered = (
                    zone_left < cx < zone_right and
                    zone_top < cy < zone_bottom
                )

                color = (0,255,0)

                cv2.rectangle(frame, (x,y),
                              (x+bw, y+bh),
                              color, 2)

                cv2.circle(frame, (cx,cy), 5, color, -1)

                cv2.putText(frame,
                            f"Ripe {score:.2f}",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color, 2)

                if is_centered and not triggered:
                    cv2.putText(frame, "TARGET LOCKED",
                                (center_x-80, center_y-70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,255), 2)
                    cv2.imshow("Harvest Vision", frame)
                    cv2.waitKey(1)

                    pick_and_drop()
                    triggered = True
                    break

        # -------- FPS --------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0), 2)

        cv2.imshow("Harvest Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
