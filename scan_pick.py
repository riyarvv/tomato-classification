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

print("🚀 scan_pick.py STARTED")

# ==========================================
# 📏 ARM PARAMETERS
# ==========================================
L1 = 14.5
L2 = 13.5
L3 = 9.0
BASE_HEIGHT = 6.5

FOV_H = 48.8
FOV_V = 36.6

# ==========================================
# ⚙️ SERVO CALIBRATION
# ==========================================
BASE_OFFSET = 20
SHOULDER_OFFSET = 160
ELBOW_OFFSET = 20

BASE_DIR = 1
SHOULDER_DIR = -1
ELBOW_DIR = 1

# ==========================================
# LIMITS
# ==========================================
BASE_MIN, BASE_MAX = 10, 100
SH_MIN, SH_MAX = 120, 160
EL_MIN, EL_MAX = 20, 65

# ==========================================
# HARDWARE INIT
# ==========================================
print("🔧 Initializing hardware...")

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

print("✅ Hardware initialized")

# ==========================================
# SMOOTH MOVEMENT
# ==========================================
def move_smooth(ch, target, delay=0.01):
    current = servos[ch].angle or target
    print(f"➡ Moving Servo {ch} from {current} → {target}")

    step = 1 if target > current else -1

    for angle in range(int(current), int(target), step):
        servos[ch].angle = angle
        time.sleep(delay)

    servos[ch].angle = target

# ==========================================
# DISTANCE
# ==========================================
def get_distance():
    vals = []
    for _ in range(5):
        vals.append(sensor.distance * 100)
        time.sleep(0.05)

    dist = sum(vals)/len(vals)
    print(f"📏 Distance: {dist:.2f} cm")
    return dist

# ==========================================
# PIXEL → WORLD
# ==========================================
def pixel_to_world(cx, cy, w, h, Z):

    angle_x = (cx - w/2) / w * math.radians(FOV_H)
    angle_y = (cy - h/2) / h * math.radians(FOV_V)

    X = Z * math.tan(angle_x)
    Y = Z * math.tan(angle_y)

    print(f"🌍 World Coords → X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}")

    return X, Y

# ==========================================
# IK
# ==========================================
def inverse_kinematics(X, Y, Z):

    print("🔄 Running IK...")

    r = math.sqrt(X**2 + Z**2)
    h = BASE_HEIGHT - Y

    D = (r**2 + h**2 - L1**2 - L2**2)/(2*L1*L2)

    if abs(D) > 1:
        print("❌ IK Failed (Out of Reach)")
        return None

    theta2 = math.acos(D)
    theta1 = math.atan2(h, r) - math.atan2(
        L2 * math.sin(theta2),
        L1 + L2 * math.cos(theta2)
    )

    base = math.degrees(math.atan2(X, Z))
    shoulder = math.degrees(theta1)
    elbow = math.degrees(theta2)

    print(f"🧠 IK Angles → Base:{base:.2f}, Shoulder:{shoulder:.2f}, Elbow:{elbow:.2f}")

    return base, shoulder, elbow

# ==========================================
# SERVO MAP
# ==========================================
def to_servo_angles(base, shoulder, elbow):

    base_s = BASE_OFFSET + BASE_DIR * base
    shoulder_s = SHOULDER_OFFSET + SHOULDER_DIR * shoulder
    elbow_s = ELBOW_OFFSET + ELBOW_DIR * elbow

    base_s = int(max(BASE_MIN, min(BASE_MAX, base_s)))
    shoulder_s = int(max(SH_MIN, min(SH_MAX, shoulder_s)))
    elbow_s = int(max(EL_MIN, min(EL_MAX, elbow_s)))

    print(f"🎯 Servo Angles → B:{base_s}, S:{shoulder_s}, E:{elbow_s}")

    return base_s, shoulder_s, elbow_s

# ==========================================
# HOME
# ==========================================
def go_home():
    print("🏠 Moving to HOME position")

    servos[BASE_CH].angle = BASE_OFFSET
    servos[SHOULDER_CH].angle = SHOULDER_OFFSET
    servos[ELBOW_CH].angle = ELBOW_OFFSET
    servos[GRIPPER_CH].angle = 100

# ==========================================
# PICK
# ==========================================
def move_to_target(cx, cy, frame):

    print("🎯 TARGET LOCKED")

    h, w = frame.shape[:2]

    Z = get_distance()
    X, Y = pixel_to_world(cx, cy, w, h, Z)

    angles = inverse_kinematics(X, Y, Z)

    if angles is None:
        return

    base, shoulder, elbow = angles

    base_s, shoulder_s, elbow_s = to_servo_angles(base, shoulder, elbow)

    move_smooth(BASE_CH, base_s)
    move_smooth(SHOULDER_CH, shoulder_s)
    move_smooth(ELBOW_CH, elbow_s)

    time.sleep(1)

    print("🤏 Closing gripper")
    servos[GRIPPER_CH].angle = 15
    time.sleep(1)

    try:
        requests.get("http://raspberrypi.local:5000/increment")
        print("📡 Count updated")
    except:
        print("⚠️ Server update failed")

    go_home()

# ==========================================
# YOLO LOAD
# ==========================================
print("🧠 Loading YOLO model...")

interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

print("✅ YOLO Loaded")

# ==========================================
# CAMERA
# ==========================================
print("📷 Opening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ CAMERA FAILED TO OPEN")
else:
    print("✅ Camera started")

CONF = 0.3
RIPE_ID = 2

# ==========================================
# START
# ==========================================
go_home()
time.sleep(2)

print("🔁 Entering main loop...")

while True:

    print("🔄 Loop running...")

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame capture failed")
        break

    print("✅ Frame captured")

    img = cv2.resize(frame, (inp[0]['shape'][2], inp[0]['shape'][1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    img = np.expand_dims(img, axis=0)

    print("🧠 Running YOLO...")

    interpreter.set_tensor(inp[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(out[0]['index'])[0].T

    detected = False

    for pred in output:

        x,y,w,h = pred[:4]
        scores = pred[4:]

        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > 0.3:
            print(f"🔍 Detected class {cid} with confidence {conf:.2f}")

        if conf > CONF and cid == RIPE_ID:

            detected = True
            print("🍅 RIPE TOMATO DETECTED")

            cx = int(x * frame.shape[1])
            cy = int(y * frame.shape[0])

            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

            move_to_target(cx, cy, frame)
            time.sleep(2)
            break

    if not detected:
        print("❌ No ripe tomato detected")

    cv2.imshow("Tomato Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pca.deinit()

print("🛑 Program ended")
