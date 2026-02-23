import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. ARM INITIALIZATION
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH = 0, 1, 2, 3, 5

servos = {}
channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH]
for ch in channels:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

LIMITS = {
    BASE_CH:     {"neutral": 20,  "min": 10,  "max": 50},
    PITCH_CH:    {"neutral": 90,  "min": 40,  "max": 120},
    SHOULDER_CH: {"neutral": 125, "pick": 115},
    ELBOW_CH:    {"neutral": 30,  "pick": 50},
    GRIPPER_CH:  {"open": 170,    "close": 20}
}

def move_slow(channel_id, target_angle, speed=0.04):
    current = servos[channel_id].angle
    if current is None:
        current = 90
    start_angle = int(current)
    target_angle = int(max(0, min(180, target_angle)))

    if start_angle == target_angle:
        return

    step = 1 if target_angle > start_angle else -1
    for angle in range(start_angle, target_angle + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

def go_home():
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_and_drop():
    move_slow(BASE_CH, 40)
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], speed=0.02)
    time.sleep(1)
    servos[GRIPPER_CH].angle = None

    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], speed=0.02)

    go_home()

# ==========================================
# 2. TFLITE INITIALIZATION
# ==========================================
MODEL_PATH = "tomato_model_pi_v11.tflite"

interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
HEALTHY_CLASS_INDEX = 1

cap = cv2.VideoCapture(0)
go_home()

# ==========================================
# 3. MAIN LOOP
# ==========================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        zone_size = 100
        zone_left = center_x - (zone_size // 2)
        zone_right = center_x + (zone_size // 2)
        zone_top = center_y - (zone_size // 2)
        zone_bottom = center_y + (zone_size // 2)

        # Draw crosshair
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (255, 255, 255), 1)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red mask
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask_red = mask_red1 + mask_red2

        # Morphological filtering (noise removal)
        kernel = np.ones((5,5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        # ------------------------------------------
        # CHECK IF ENTIRE TARGET ZONE IS RED
        # ------------------------------------------
        zone_mask = mask_red[zone_top:zone_bottom, zone_left:zone_right]
        red_pixels = cv2.countNonZero(zone_mask)
        total_pixels = zone_mask.shape[0] * zone_mask.shape[1]
        red_ratio = red_pixels / total_pixels

        is_fully_red = red_ratio > 0.90  # 90% coverage required

        # Find contours (for classification & drawing)
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            tomato_crop = frame[y:y+h, x:x+w]
            if tomato_crop.size == 0:
                continue

            img = cv2.resize(tomato_crop, (224, 224)).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

            if class_idx == HEALTHY_CLASS_INDEX and confidence >= 0.60:
                label = "Healthy"
                color = (0, 255, 0)

                if is_fully_red:
                    cv2.putText(frame, "TARGET LOCKED", (center_x - 70, center_y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Harvest Vision", frame)
                    cv2.waitKey(1)

                    pick_and_drop()
                    break
                else:
                    cv2.putText(frame, "ALIGNING...", (x, y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                label = "Unhealthy"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Ripe {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Harvest Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
