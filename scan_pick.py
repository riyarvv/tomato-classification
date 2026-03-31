import time
import cv2
import numpy as np
import math
import board
import busio

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor

# ==========================================
# 📏 ARM PARAMETERS
# ==========================================
L1, L2 = 14.5, 13.5
BASE_HEIGHT = 6.5

# ==========================================
# 🎯 SERVO SETTINGS
# ==========================================
BASE_OFFSET = 20
SHOULDER_OFFSET = 160
ELBOW_OFFSET = 20

BASE_DIR = 1
SHOULDER_DIR = -1
ELBOW_DIR = 1

BASE_MIN, BASE_MAX = 10, 100
SH_MIN, SH_MAX = 120, 160
EL_MIN, EL_MAX = 20, 65

# ==========================================
# 🔌 HARDWARE INIT
# ==========================================
sensor = DistanceSensor(echo=24, trigger=23)

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

BASE_CH, SHOULDER_CH, ELBOW_CH, GRIPPER_CH = 0,1,2,5

servos = {
    BASE_CH: servo.Servo(pca.channels[BASE_CH]),
    SHOULDER_CH: servo.Servo(pca.channels[SHOULDER_CH]),
    ELBOW_CH: servo.Servo(pca.channels[ELBOW_CH]),
    GRIPPER_CH: servo.Servo(pca.channels[GRIPPER_CH]),
}

# ==========================================
# 🎥 CAMERA (LOW LAG)
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(3, 160)
cap.set(4, 120)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==========================================
# 🧠 MODEL
# ==========================================
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

CONF = 0.7   # higher confidence
RIPE_ID = 2

# ==========================================
# ⚙️ FUNCTIONS
# ==========================================
def move_smooth(ch, target, delay=0.01):
    current = servos[ch].angle or target
    step = 1 if target > current else -1
    for angle in range(int(current), int(target), step):
        servos[ch].angle = angle
        time.sleep(delay)

def gripper_close():
    for a in [100,90,75,60,45,30,15]:
        servos[GRIPPER_CH].angle = a
        time.sleep(0.3)

def gripper_open():
    for a in [15,30,45,60,75,90,100]:
        servos[GRIPPER_CH].angle = a
        time.sleep(0.3)

def get_distance():
    vals = []
    for _ in range(5):
        d = sensor.distance * 100
        if 2 < d < 200:
            vals.append(d)
        time.sleep(0.05)
    return sum(vals)/len(vals) if vals else 25

# ==========================================
# 🎯 VISUAL SERVO
# ==========================================
def visual_servo(X, Y):
    Kx = 0.05
    Ky = 0.05

    base = servos[BASE_CH].angle or 20
    shoulder = servos[SHOULDER_CH].angle or 160

    base += -X * Kx
    shoulder += -Y * Ky

    base = max(BASE_MIN, min(BASE_MAX, base))
    shoulder = max(SH_MIN, min(SH_MAX, shoulder))

    servos[BASE_CH].angle = base
    servos[SHOULDER_CH].angle = shoulder

# ==========================================
# 🤖 IK
# ==========================================
def inverse_kinematics(Z):
    r = Z
    h = BASE_HEIGHT

    D = (r**2 + h**2 - L1**2 - L2**2)/(2*L1*L2)
    if abs(D) > 1:
        return None

    theta2 = math.acos(D)
    theta1 = math.atan2(h, r) - math.atan2(
        L2 * math.sin(theta2),
        L1 + L2 * math.cos(theta2)
    )

    shoulder = math.degrees(theta1)
    elbow = math.degrees(theta2)

    return shoulder, elbow

# ==========================================
# 🏠 HOME (IMPORTANT FIX)
# ==========================================
def go_home():
    move_smooth(BASE_CH, 20)
    move_smooth(SHOULDER_CH, 160)
    move_smooth(ELBOW_CH, 20)
    servos[GRIPPER_CH].angle = 100

go_home()
time.sleep(2)

# ==========================================
# 🔁 LOOP CONTROL
# ==========================================
frame_count = 0
locked_frames = 0
movement_enabled = False
Z = 25

# ==========================================
# 🔁 MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames (reduce lag)
    if frame_count % 3 != 0:
        cv2.imshow("Tomato Bot", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    h_img, w_img = frame.shape[:2]
    cx0, cy0 = w_img//2, h_img//2

    # Draw axes
    cv2.line(frame, (cx0, 0), (cx0, h_img), (255,0,0), 1)
    cv2.line(frame, (0, cy0), (w_img, cy0), (255,0,0), 1)

    # Run YOLO less frequently
    if frame_count % 6 != 0:
        cv2.imshow("Tomato Bot", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # Preprocess
    resized = cv2.resize(frame, (inp[0]['shape'][2], inp[0]['shape'][1]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)/255
    rgb = np.expand_dims(rgb, axis=0)

    interpreter.set_tensor(inp[0]['index'], rgb)
    interpreter.invoke()

    output = interpreter.get_tensor(out[0]['index'])[0].T

    for pred in output:
        x,y,w,h = pred[:4]
        scores = pred[4:]
        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > CONF and cid == RIPE_ID:

            cx = int(x * w_img)
            cy = int(y * h_img)

            cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            X = cx - cx0
            Y = cy0 - cy

            # Update Z less frequently
            if frame_count % 10 == 0:
                Z = get_distance()

            cv2.putText(frame,f"X:{X} Y:{Y} Z:{Z:.1f}",
                        (cx+10,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

            # Stability check
            if abs(X) < 100 and abs(Y) < 100:
                locked_frames += 1
            else:
                locked_frames = 0

            # Enable movement only after stable detection
            if locked_frames > 5:
                movement_enabled = True

            # ALIGN ONLY AFTER LOCK
            if movement_enabled:
                if abs(X) > 20 or abs(Y) > 20:
                    visual_servo(X, Y)

                else:
                    print("✅ LOCKED & PICKING")

                    target_Z = Z - 5
                    ik = inverse_kinematics(target_Z)

                    if ik:
                        sh, el = ik

                        sh = SHOULDER_OFFSET + SHOULDER_DIR * sh
                        el = ELBOW_OFFSET + ELBOW_DIR * el

                        sh = max(SH_MIN, min(SH_MAX, sh))
                        el = max(EL_MIN, min(EL_MAX, el))

                        move_smooth(SHOULDER_CH, sh)
                        move_smooth(ELBOW_CH, el)

                        gripper_close()
                        time.sleep(1)

                        go_home()
                        gripper_open()

                        movement_enabled = False
                        locked_frames = 0
                        time.sleep(2)

            break

    cv2.imshow("Tomato Bot", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
