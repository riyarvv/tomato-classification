import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1️⃣ PCA9685 INITIALIZATION
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==========================================
# 2️⃣ CHANNEL MAPPING
# ==========================================
BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH = 0,1,2,3,5,6

channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH]
servos = {}

for ch in channels:
    servos[ch] = servo.Servo(
        pca.channels[ch],
        min_pulse=500,
        max_pulse=2500
    )

# ==========================================
# 3️⃣ LIMITS
# ==========================================
LIMITS = {
    BASE_CH:     {"neutral":20, "min":10, "max":100},
    SHOULDER_CH: {"neutral":130, "pick":115},
    ELBOW_CH:    {"neutral":30,  "pick":50},
    PITCH_CH:    {"neutral":90},
    GRIPPER_CH:  {"open":170, "close":20},
    CAMERA_CH:   {"min":10, "max":100}
}

# ==========================================
# 4️⃣ SMOOTH MOVEMENT
# ==========================================
def move_slow(channel, target, delay=0.02):
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
# 5️⃣ PICK FUNCTION
# ==========================================
def pick_tomato():
    print("🤖 Picking tomato...")
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], delay=0.01)
    time.sleep(1)

def drop_tomato():
    print("📦 Moving to drop position...")
    move_slow(BASE_CH, 10, delay=0.02)   # Move base to 10°
    servos[CAMERA_CH].angle = servos[BASE_CH].angle
    time.sleep(0.5)

    print("🪴 Dropping tomato...")
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], delay=0.01)

# ==========================================
# 6️⃣ LOAD MODEL
# ==========================================
MODEL_PATH = "tomato_model_pi_v11.tflite"
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
HEALTHY_CLASS_INDEX = 1

# ==========================================
# 7️⃣ CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

# ==========================================
# 8️⃣ SCANNING VARIABLES
# ==========================================
scan_angle = LIMITS[BASE_CH]["min"]
scan_direction = 1
locked = False

try:
    while True:

        # 🔄 SCANNING
        if not locked:
            scan_angle += scan_direction

            if scan_angle >= LIMITS[BASE_CH]["max"] or scan_angle <= LIMITS[BASE_CH]["min"]:
                scan_direction *= -1

            move_slow(BASE_CH, scan_angle, delay=0.01)
            servos[CAMERA_CH].angle = servos[BASE_CH].angle

        # 📷 FRAME
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        center_x, center_y = width//2, height//2

        # 🎯 Draw CROSSHAIR "+"
        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (255,255,255), 2)
        cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (255,255,255), 2)

        zone_size = 100
        zone_left = center_x - zone_size//2
        zone_right = center_x + zone_size//2
        zone_top = center_y - zone_size//2
        zone_bottom = center_y + zone_size//2

        cv2.rectangle(frame,(zone_left,zone_top),(zone_right,zone_bottom),(255,255,255),1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
                   cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))

        contours,_ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue

            x,y,w,h = cv2.boundingRect(cnt)
            tomato_center_x = x + w//2
            tomato_center_y = y + h//2

            tomato_crop = frame[y:y+h, x:x+w]
            if tomato_crop.size == 0:
                continue

            img = cv2.resize(tomato_crop,(224,224)).astype(np.float32)/255.0
            img = np.expand_dims(img,axis=0)

            interpreter.set_tensor(input_details[0]['index'],img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]

            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

            is_centered = (zone_left < tomato_center_x < zone_right) and \
                          (zone_top < tomato_center_y < zone_bottom)

            if class_idx == HEALTHY_CLASS_INDEX and confidence >= 0.60:

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                if is_centered and not locked:
                    print(f"🎯 Target locked at angle {scan_angle}")
                    locked = True

                    pick_tomato()
                    drop_tomato()

                    print("🛑 Task Complete. Stopping system.")
                    cap.release()
                    cv2.destroyAllWindows()
                    pca.deinit()
                    exit()

        cv2.imshow("Harvest Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
