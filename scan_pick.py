import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1️⃣ MODEL SETTINGS
# ==========================================
MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.5

# ==========================================
# 2️⃣ SERVO CHANNELS
# ==========================================
BASE_CH = 0
SHOULDER_CH = 1
ELBOW_CH = 2
GRIPPER_CH = 3
CAMERA_CH = 4

# ==========================================
# 3️⃣ SERVO LIMITS
# ==========================================
LIMITS = {
    BASE_CH: {"min": 0, "max": 180, "neutral": 90},
    SHOULDER_CH: {"min": 40, "max": 140, "neutral": 90, "pick": 130},
    ELBOW_CH: {"min": 40, "max": 140, "neutral": 90, "pick": 60},
    GRIPPER_CH: {"min": 15, "max": 120, "neutral": 15},
    CAMERA_CH: {"min": 0, "max": 180, "neutral": 90},
}

# ==========================================
# 4️⃣ INITIALIZE PCA9685
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

servos = [servo.Servo(pca.channels[i]) for i in range(5)]

# ==========================================
# 5️⃣ INITIAL POSITIONS
# ==========================================
for ch in LIMITS:
    servos[ch].angle = LIMITS[ch]["neutral"]

# ==========================================
# 6️⃣ LOAD MODEL
# ==========================================
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, in_h, in_w, _ = input_details[0]["shape"]

# ==========================================
# 7️⃣ HELPER FUNCTIONS
# ==========================================

def move_slow(channel, target, delay=0.01):
    current = int(servos[channel].angle)
    step = 1 if target > current else -1
    for angle in range(current, target + step, step):
        servos[channel].angle = angle
        time.sleep(delay)


def close_gripper(delay=1.5):  # Arduino-like slow
    sequence = [15, 30, 45, 60, 75, 90, 100, 120]
    for angle in sequence:
        servos[GRIPPER_CH].angle = angle
        time.sleep(delay)


def open_gripper(delay=1.5):
    sequence = [120, 100, 90, 75, 60, 45, 30, 15]
    for angle in sequence:
        servos[GRIPPER_CH].angle = angle
        time.sleep(delay)


def move_base_camera_slow(target):
    current = int(servos[BASE_CH].angle)
    step = 1 if target > current else -1
    for angle in range(current, target + step, step):
        servos[BASE_CH].angle = angle
        servos[CAMERA_CH].angle = angle
        time.sleep(0.03)


def pick_and_drop():
    print("🍅 Picking...")

    # Move arm down
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"], 0.01)
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"], 0.01)

    # Close gripper slowly
    close_gripper()

    print("⬆ Lifting...")

    # Lift back to neutral
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"], 0.01)
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"], 0.01)

    print("🔄 Moving to drop position...")

    # Move base + camera to neutral very slowly
    move_base_camera_slow(LIMITS[BASE_CH]["neutral"])

    print("📦 Dropping slowly...")

    open_gripper()

    print("✅ Done")

# ==========================================
# 8️⃣ CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

scan_angle = LIMITS[BASE_CH]["neutral"]
scan_direction = 1
locked = False

print("🚀 Starting scan...")

# ==========================================
# 9️⃣ MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Preprocess
    resized = cv2.resize(frame, (in_w, in_h))
    input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    boxes = []
    scores = []
    centers = []

    for detection in output_data:
        conf = detection[4]
        if conf > CONF_THRESHOLD:
            cx = int(detection[0] * frame_w)
            cy = int(detection[1] * frame_h)
            bw = int(detection[2] * frame_w)
            bh = int(detection[3] * frame_h)

            x = int(cx - bw / 2)
            y = int(cy - bh / 2)

            boxes.append([x, y, bw, bh])
            scores.append(float(conf))
            centers.append((cx, cy))

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, 0.4)

    # Draw detection zone
    zone_left = frame_w // 2 - 50
    zone_right = frame_w // 2 + 50
    zone_top = frame_h // 2 - 50
    zone_bottom = frame_h // 2 + 50

    cv2.rectangle(frame, (zone_left, zone_top),
                  (zone_right, zone_bottom), (255, 0, 0), 2)

    if len(indices) > 0:
        for idx in indices.flatten():

            x, y, bw, bh = boxes[idx]
            score = scores[idx]
            cx, cy = centers[idx]

            # Draw box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                          (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Ripe {score:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            is_centered = (
                zone_left < cx < zone_right and
                zone_top < cy < zone_bottom
            )

            if is_centered and not locked:
                print(f"🎯 Locked at {scan_angle}")
                locked = True

                pick_and_drop()

                scan_angle = LIMITS[BASE_CH]["neutral"]
                locked = False
                break

    # SCANNING (only if not locked)
    if not locked:
        servos[BASE_CH].angle = scan_angle
        servos[CAMERA_CH].angle = scan_angle

        scan_angle += scan_direction

        if scan_angle >= LIMITS[BASE_CH]["max"] or \
           scan_angle <= LIMITS[BASE_CH]["min"]:
            scan_direction *= -1

    cv2.imshow("Tomato Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
