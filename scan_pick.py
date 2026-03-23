import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor

# ==============================
# SENSOR
# ==============================
sensor = DistanceSensor(echo=24, trigger=23)

# ==============================
# ARM PARAMETERS
# ==============================
L1 = 14.50
L2 = 13.50
L3 = 9.0
BASE_HEIGHT = 6.50

BASE_MIN, BASE_MAX = 10, 100
SHOULDER_NEUTRAL = 160
SHOULDER_DOWN = 120
ELBOW_NEUTRAL = 20
ELBOW_MAX = 65

FOV_H = 48.8
FOV_V = 36.6

# ==============================
# PCA9685
# ==============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

BASE_CH, SHOULDER_CH, ELBOW_CH, GRIPPER_CH = 0,1,2,5

servos = {}
for ch in [BASE_CH, SHOULDER_CH, ELBOW_CH, GRIPPER_CH]:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# ==============================
# SMOOTH MOVE
# ==============================
def move_smooth(ch, target, step=1, delay=0.01):
    current = servos[ch].angle or target
    current = int(current)
    target = int(target)

    if current < target:
        rng = range(current, target, step)
    else:
        rng = range(current, target, -step)

    for a in rng:
        servos[ch].angle = a
        time.sleep(delay)

    servos[ch].angle = target

# ==============================
# SLOW GRIPPER (YOUR VERSION)
# ==============================
def gripper_open_slow():
    steps = [15, 30, 45, 60, 75, 90, 100]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)

def gripper_close_slow():
    steps = [100, 90, 75, 60, 45, 30, 15]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)

# ==============================
# DISTANCE
# ==============================
def get_distance():
    vals = []
    for _ in range(5):
        vals.append(sensor.distance * 100)
        time.sleep(0.02)
    return sum(vals)/len(vals)

# ==============================
# PIXEL → ANGLE
# ==============================
def pixel_to_angle(cx, cy, w, h):
    dx = cx - w/2
    dy = cy - h/2

    ax = (dx / w) * FOV_H
    ay = (dy / h) * FOV_V

    return np.radians(ax), np.radians(ay)

# ==============================
# ANGLE → XYZ
# ==============================
def get_xyz(cx, cy, w, h, dist):
    ax, ay = pixel_to_angle(cx, cy, w, h)

    Z = dist - 2   # calibration
    X = Z * np.tan(ax)
    Y = Z * np.tan(ay)

    return X, Y, Z

# ==============================
# INVERSE KINEMATICS
# ==============================
def inverse_kinematics(X, Y, Z):

    # BASE
    base = np.degrees(np.arctan2(X, Z))
    base = 55 + base   # center offset

    # PLANAR
    r = np.sqrt(X**2 + Z**2)
    y = Y - BASE_HEIGHT

    D = np.sqrt(r**2 + y**2)

    # ELBOW
    cos_elbow = (D**2 - L1**2 - L2**2)/(2*L1*L2)
    cos_elbow = np.clip(cos_elbow, -1, 1)
    elbow = np.degrees(np.arccos(cos_elbow))

    # SHOULDER
    alpha = np.arctan2(y, r)
    beta = np.arccos((L1**2 + D**2 - L2**2)/(2*L1*D))
    shoulder = np.degrees(alpha + beta)

    # 🔴 MAP TO YOUR SERVO RANGE
    shoulder = SHOULDER_NEUTRAL - shoulder
    elbow = ELBOW_NEUTRAL + elbow

    return base, shoulder, elbow

# ==============================
# MOVE TO XYZ
# ==============================
def move_to_xyz(X, Y, Z):

    base, shoulder, elbow = inverse_kinematics(X, Y, Z)

    base = max(BASE_MIN, min(BASE_MAX, base))
    shoulder = max(SHOULDER_DOWN, min(SHOULDER_NEUTRAL, shoulder))
    elbow = max(ELBOW_NEUTRAL, min(ELBOW_MAX, elbow))

    print(f"Angles → B:{base:.1f} S:{shoulder:.1f} E:{elbow:.1f}")

    move_smooth(BASE_CH, base, step=1, delay=0.01)
    move_smooth(SHOULDER_CH, shoulder, step=1, delay=0.01)
    move_smooth(ELBOW_CH, elbow, step=1, delay=0.01)

# ==============================
# PICK FUNCTION
# ==============================
def pick(X, Y, Z):

    print("🍅 Picking...")

    move_to_xyz(X, Y, Z + 5)   # approach
    move_to_xyz(X, Y, Z)       # final

    gripper_close_slow()

    import requests

    try:
        requests.get("http://raspberrypi.local:5000/increment")
    except:
        pass

    move_to_xyz(X, Y, Z + 8)   # lift

    gripper_open_slow()

    print("✅ Done")

# ==============================
# YOLO
# ==============================
MODEL_PATH = "/home/rslvpi5/tomato-detection/tomato-classification/best_float16.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

H, W = input_details[0]['shape'][1:3]

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

# ==============================
# INITIAL POSITION
# ==============================
servos[BASE_CH].angle = 55
servos[SHOULDER_CH].angle = SHOULDER_NEUTRAL
servos[ELBOW_CH].angle = ELBOW_NEUTRAL
gripper_open_slow()

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    img = cv2.resize(frame, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0].T

    for pred in output:

        x, y, bw, bh = pred[:4]
        scores = pred[4:]
        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > 0.4 and cid == 2:

            cx = int(x * w)
            cy = int(y * h)

            dist = get_distance()

            X, Y, Z = get_xyz(cx, cy, w, h, dist)

            print(f"XYZ → {X:.2f}, {Y:.2f}, {Z:.2f}")

            pick(X, Y, Z)

            time.sleep(2)
            break

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
pca.deinit()
cv2.destroyAllWindows()
