import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# PCA9685 INITIALIZATION
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==========================================
# CHANNEL MAPPING
# ==========================================
BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH = 0,1,2,3,5,6
channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, GRIPPER_CH, CAMERA_CH]

servos = {}
for ch in channels:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# ==========================================
# SERVO LIMITS
# ==========================================
LIMITS = {
    BASE_CH:     {"neutral":20, "min":10, "max":100},
    SHOULDER_CH: {"neutral":130, "pick":115},
    ELBOW_CH:    {"neutral":30,  "pick":50},
    GRIPPER_CH:  {"open":170, "close":20},
}

# ==========================================
# SET INITIAL NEUTRAL POSITION
# ==========================================
servos[BASE_CH].angle = LIMITS[BASE_CH]["neutral"]
servos[SHOULDER_CH].angle = LIMITS[SHOULDER_CH]["neutral"]
servos[ELBOW_CH].angle = LIMITS[ELBOW_CH]["neutral"]
servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]
servos[CAMERA_CH].angle = LIMITS[BASE_CH]["neutral"]
time.sleep(2)

# ==========================================
# SMOOTH MOVEMENT
# ==========================================
def move_slow(channel, target, delay=0.03):
    current = servos[channel].angle
    if current is None:
        current = target
    current = int(current)
    target = int(target)
    step = 1 if target > current else -1

    for angle in range(current, target + step, step):
        servos[channel].angle = angle
        time.sleep(delay)

# ==========================================
# PICK + DROP SEQUENCE
# ==========================================
def pick_and_drop():
    print("🤖 Picking tomato...")

    # Move to pick position
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], delay=0.01)
    time.sleep(1)

    # Return arm to neutral BEFORE rotating base
    print("↩ Returning arm to neutral")
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])

    # Rotate base to drop position
    print("📦 Moving base to drop position")
    move_slow(BASE_CH, 10, delay=0.03)
    servos[CAMERA_CH].angle = 10
    time.sleep(0.5)

    # Open gripper to drop
    print("🪴 Dropping tomato...")
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], delay=0.01)

    print("✅ Task complete. Stopping.")
    time.sleep(2)

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = "tomato_model_pi_v11.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

HEALTHY_CLASS_INDEX = 0   # adjust if needed
MIN_CONFIDENCE = 75
REQUIRED_STABLE_FRAMES = 5
stable_frames = 0

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

# ==========================================
# SCANNING VARIABLES
# ==========================================
scan_angle = LIMITS[BASE_CH]["neutral"]
scan_direction = 1

# ==========================================
# MAIN LOOP
# ==========================================
try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # HSV RED DETECTION
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([10,255,255])
        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([180,255,255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + \
                   cv2.inRange(hsv, lower_red2, upper_red2)

        kernel = np.ones((5,5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel)

        contours,_ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ======================================
        # SLOW SCANNING IF NO TOMATO
        # ======================================
        if len(contours) == 0:
            scan_angle += scan_direction

            if scan_angle >= LIMITS[BASE_CH]["max"] or scan_angle <= LIMITS[BASE_CH]["min"]:
                scan_direction *= -1

            servos[BASE_CH].angle = scan_angle
            servos[CAMERA_CH].angle = scan_angle
            time.sleep(0.02)

        # ======================================
        # PROCESS DETECTIONS
        # ======================================
        for cnt in contours:

            if cv2.contourArea(cnt) < 1000:
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            img = cv2.resize(crop,(224,224)).astype(np.float32)/255.0
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx] * 100

            print("Confidence:", confidence)

            if (class_idx == HEALTHY_CLASS_INDEX and
                confidence >= MIN_CONFIDENCE):

                stable_frames += 1
                print("Stable frames:", stable_frames)

                if stable_frames >= REQUIRED_STABLE_FRAMES:
                    print("🎯 FINAL LOCK")
                    pick_and_drop()
                    cap.release()
                    cv2.destroyAllWindows()
                    pca.deinit()
                    exit()
            else:
                stable_frames = 0

            color = (0,255,0) if class_idx == HEALTHY_CLASS_INDEX else (0,0,255)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.imshow("AI Harvest System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
