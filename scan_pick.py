import time
import board
import busio
import cv2
import numpy as np
import requests

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor

print("🚀 scan_pick.py STARTED")

# ==========================================
# SAFE SERVO POSITIONS (YOUR CALIBRATION)
# ==========================================
BASE_OFFSET = 20
SHOULDER_OFFSET = 160
ELBOW_OFFSET = 20

BASE_MIN, BASE_MAX = 10, 100

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
# SMOOTH MOVEMENT
# ==========================================
def move_smooth(ch, target, delay=0.03):
    current = servos[ch].angle or target
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
    return sum(vals)/len(vals)

# ==========================================
# HOME POSITION
# ==========================================
def go_home():
    print("🏠 Going HOME")

    servos[BASE_CH].angle = BASE_OFFSET
    time.sleep(0.3)

    servos[SHOULDER_CH].angle = SHOULDER_OFFSET
    time.sleep(0.3)

    servos[ELBOW_CH].angle = ELBOW_OFFSET
    time.sleep(0.3)

    servos[GRIPPER_CH].angle = 100

# ==========================================
# ALIGN FUNCTION (VISION CONTROL)
# ==========================================
def align_to_tomato(cx, cy, w, h):

    center_x = w // 2
    center_y = int(h * 0.7)

    error_x = cx - center_x
    error_y = cy - center_y

    # BASE CONTROL (LEFT/RIGHT)
    base = servos[BASE_CH].angle + int(error_x * 0.05)
    base = max(BASE_MIN, min(BASE_MAX, base))
    servos[BASE_CH].angle = base

    # SHOULDER CONTROL (UP/DOWN SAFE)
    shoulder = servos[SHOULDER_CH].angle - int(error_y * 0.03)
    shoulder = max(130, min(170, shoulder))
    servos[SHOULDER_CH].angle = shoulder

    if abs(error_x) < 25 and abs(error_y) < 25:
        return True

    return False

# ==========================================
# PICK FUNCTION (SAFE)
# ==========================================
def pick_tomato():

    print("➡ Moving forward")

    while True:

        dist = get_distance()
        print(f"📏 Distance: {dist:.2f} cm")

        if dist < 6:
            print("📍 Close enough")
            break

        current = servos[ELBOW_CH].angle

        if current >= 60:
            print("⚠️ Max reach")
            break

        move_smooth(ELBOW_CH, current + 2)

    time.sleep(0.5)

    print("🤏 Closing gripper")
    servos[GRIPPER_CH].angle = 15
    time.sleep(1)

    # UPDATE COUNT
    try:
        requests.get("http://raspberrypi.local:5000/increment")
        print("📡 Count updated")
    except:
        print("⚠️ Server not reachable")

    print("↩ Returning HOME")

    move_smooth(ELBOW_CH, ELBOW_OFFSET)
    move_smooth(SHOULDER_CH, SHOULDER_OFFSET)
    move_smooth(BASE_CH, BASE_OFFSET)

    servos[GRIPPER_CH].angle = 100

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

CONF = 0.6
RIPE_ID = 2

# ==========================================
# START
# ==========================================
go_home()
time.sleep(2)

lock_counter = 0

print("🔁 Running...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    img = cv2.resize(frame, (inp[0]['shape'][2], inp[0]['shape'][1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(inp[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(out[0]['index'])[0].T

    detected = False

    for pred in output:

        x,y,w_box,h_box = pred[:4]
        scores = pred[4:]

        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > CONF and cid == RIPE_ID:

            detected = True

            cx = int(x * frame.shape[1])
            cy = int(y * frame.shape[0])

            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

            aligned = align_to_tomato(cx, cy, frame.shape[1], frame.shape[0])

            if aligned:
                lock_counter += 1
            else:
                lock_counter = 0

            if lock_counter > 5:
                print("🎯 LOCKED → PICKING")
                pick_tomato()
                lock_counter = 0
                break

    if not detected:
        lock_counter = 0

    cv2.imshow("Tomato Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pca.deinit()

print("🛑 Program ended")
