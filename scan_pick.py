import time
import board
import busio
import cv2
import numpy as np
import math
import requests

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor

# ==========================================
# 📏 ARM PARAMETERS
# ==========================================
L1 = 14.5
L2 = 13.5
BASE_HEIGHT = 6.5

FOV_H = 48.8
FOV_V = 36.6

# ==========================================
# ✅ CORRECT NEUTRAL POSITIONS (YOUR VALUES)
# ==========================================
BASE_OFFSET = 20
SHOULDER_OFFSET = 160
ELBOW_OFFSET = 20

# Start with all directions positive
BASE_DIR = 1
SHOULDER_DIR = 1
ELBOW_DIR = 1

# ==========================================
# LIMITS (SAFE RANGE)
# ==========================================
BASE_MIN, BASE_MAX = 10, 100
SH_MIN, SH_MAX = 120, 170
EL_MIN, EL_MAX = 20, 65

# ==========================================
# HARDWARE INIT
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
# 🔧 SMOOTH MOVE (SLOW & SAFE)
# ==========================================
def move_smooth(ch, target, delay=0.02):
    current = servos[ch].angle
    if current is None:
        current = target

    step = 1 if target > current else -1

    for angle in range(int(current), int(target), step):
        servos[ch].angle = angle
        time.sleep(delay)

# ==========================================
# 📏 DISTANCE
# ==========================================
def get_distance():
    vals = []
    for _ in range(5):
        vals.append(sensor.distance * 100)
        time.sleep(0.05)
    return sum(vals)/len(vals)

# ==========================================
# 🎯 PIXEL → WORLD
# ==========================================
def pixel_to_world(cx, cy, w, h, Z):
    angle_x = (cx - w/2) / w * math.radians(FOV_H)
    angle_y = (cy - h/2) / h * math.radians(FOV_V)

    X = Z * math.tan(angle_x)
    Y = Z * math.tan(angle_y)

    return X, Y

# ==========================================
# 🤖 INVERSE KINEMATICS
# ==========================================
def inverse_kinematics(X, Y, Z):

    r = math.sqrt(X**2 + Z**2)
    h = BASE_HEIGHT - Y

    D = (r**2 + h**2 - L1**2 - L2**2)/(2*L1*L2)

    if abs(D) > 1:
        return None

    theta2 = math.acos(D)
    theta1 = math.atan2(h, r) - math.atan2(
        L2 * math.sin(theta2),
        L1 + L2 * math.cos(theta2)
    )

    base = math.degrees(math.atan2(X, Z))
    shoulder = math.degrees(theta1)
    elbow = math.degrees(theta2)

    return base, shoulder, elbow

# ==========================================
# 🔄 CONVERT TO SERVO ANGLES
# ==========================================
def to_servo_angles(base, shoulder, elbow):

    base_s = BASE_OFFSET + BASE_DIR * base
    shoulder_s = SHOULDER_OFFSET + SHOULDER_DIR * shoulder
    elbow_s = ELBOW_OFFSET + ELBOW_DIR * elbow

    base_s = int(max(BASE_MIN, min(BASE_MAX, base_s)))
    shoulder_s = int(max(SH_MIN, min(SH_MAX, shoulder_s)))
    elbow_s = int(max(EL_MIN, min(EL_MAX, elbow_s)))

    return base_s, shoulder_s, elbow_s

# ==========================================
# 🏠 SAFE HOME POSITION
# ==========================================
def go_home():
    print("🏠 Going to safe home position")

    servos[BASE_CH].angle = BASE_OFFSET
    time.sleep(0.5)

    servos[SHOULDER_CH].angle = SHOULDER_OFFSET
    time.sleep(0.5)

    servos[ELBOW_CH].angle = ELBOW_OFFSET
    time.sleep(0.5)

    servos[GRIPPER_CH].angle = 100

# ==========================================
# 🍅 PICK FUNCTION
# ==========================================
def move_to_target(cx, cy, frame):

    h, w = frame.shape[:2]

    Z = get_distance()
    print(f"Distance: {Z:.2f} cm")

    X, Y = pixel_to_world(cx, cy, w, h, Z)
    print(f"World → X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}")

    angles = inverse_kinematics(X, Y, Z)

    if angles is None:
        print("❌ Out of reach")
        return

    base, shoulder, elbow = angles

    base_s, shoulder_s, elbow_s = to_servo_angles(base, shoulder, elbow)

    print("Servo:", base_s, shoulder_s, elbow_s)

    move_smooth(BASE_CH, base_s)
    move_smooth(SHOULDER_CH, shoulder_s)
    move_smooth(ELBOW_CH, elbow_s)

    time.sleep(1)

    # Close gripper
    servos[GRIPPER_CH].angle = 15
    time.sleep(1)

    # 🍅 Update count
    try:
        requests.get("http://raspberrypi.local:5000/increment")
        print("📡 Count updated")
    except:
        print("⚠️ Server not reachable")

    # Return home safely
    go_home()

# ==========================================
# LOAD MODEL
# ==========================================
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

CONF = 0.3
RIPE_ID = 2

# ==========================================
# START
# ==========================================
go_home()
time.sleep(2)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (inp[0]['shape'][2], inp[0]['shape'][1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(inp[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(out[0]['index'])[0].T

    for pred in output:

        x,y,w,h = pred[:4]
        scores = pred[4:]

        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > CONF and cid == RIPE_ID:

            cx = int(x * frame.shape[1])
            cy = int(y * frame.shape[0])

            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

            move_to_target(cx, cy, frame)
            time.sleep(2)
            break

    cv2.imshow("Tomato Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pca.deinit()
