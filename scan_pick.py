import time
import board
import busio
import cv2
import numpy as np
import math
import threading
import requests

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor

# ==========================================
# 📏 ARM PARAMETERS
# ==========================================
L1, L2, L3 = 14.5, 13.5, 9.0
BASE_HEIGHT = 6.5

FOV_H, FOV_V = 48.8, 36.6

# ==========================================
# 🎯 SERVO OFFSETS (FIXED HOME)
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
    BASE_CH: servo.Servo(pca.channels[BASE_CH], min_pulse=500, max_pulse=2500),
    SHOULDER_CH: servo.Servo(pca.channels[SHOULDER_CH], min_pulse=500, max_pulse=2500),
    ELBOW_CH: servo.Servo(pca.channels[ELBOW_CH], min_pulse=500, max_pulse=2500),
    GRIPPER_CH: servo.Servo(pca.channels[GRIPPER_CH], min_pulse=500, max_pulse=2500),
}

# ==========================================
# 🎥 CAMERA THREAD (SMOOTH VIDEO)
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

frame = None
lock = threading.Lock()

def camera_thread():
    global frame
    while True:
        ret, img = cap.read()
        if ret:
            with lock:
                frame = img.copy()

threading.Thread(target=camera_thread, daemon=True).start()

# ==========================================
# 📺 FLASK STREAM
# ==========================================
from flask import Flask, Response
app = Flask(__name__)

def generate():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5001, threaded=True), daemon=True).start()

# ==========================================
# ⚙️ MOVEMENT
# ==========================================
def move_smooth(ch, target, delay=0.01):
    current = servos[ch].angle or target
    step = 1 if target > current else -1

    for angle in range(int(current), int(target), step):
        servos[ch].angle = angle
        time.sleep(delay)

# ==========================================
# 🏠 HOME POSITION (FIXED)
# ==========================================
def go_home():
    print("🏠 Moving to HOME (20,160,20)")
    move_smooth(BASE_CH, 20)
    move_smooth(SHOULDER_CH, 160)
    move_smooth(ELBOW_CH, 20)
    servos[GRIPPER_CH].angle = 100
    print("✅ HOME reached")

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
# 📍 PIXEL → WORLD
# ==========================================
def pixel_to_world(cx, cy, w, h, Z):
    angle_x = (cx - w/2) / w * math.radians(FOV_H)
    angle_y = (cy - h/2) / h * math.radians(FOV_V)

    X = Z * math.tan(angle_x)
    Y = Z * math.tan(angle_y)

    return X, Y

# ==========================================
# 🤖 IK
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
# 🔄 SERVO MAP
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
# ✋ GRIPPER
# ==========================================
def gripper_close():
    for a in [100,90,75,60,45,30,15]:
        servos[GRIPPER_CH].angle = a
        time.sleep(0.3)

def gripper_open():
    for a in [15,30,45,60,75,90,100]:
        servos[GRIPPER_CH].angle = a
        time.sleep(0.3)

# ==========================================
# 🎯 MOVE TO TARGET
# ==========================================
def move_to_target(cx, cy, img):

    h, w = img.shape[:2]

    Z = get_distance()
    print(f"📏 Z: {Z:.2f} cm")

    X, Y = pixel_to_world(cx, cy, w, h, Z)
    print(f"🌍 X:{X:.2f}, Y:{Y:.2f}")

    angles = inverse_kinematics(X, Y, Z)

    if angles is None:
        print("❌ Out of reach")
        return

    base, shoulder, elbow = angles
    base_s, sh_s, el_s = to_servo_angles(base, shoulder, elbow)

    print("🎯 Servo:", base_s, sh_s, el_s)

    move_smooth(BASE_CH, base_s)
    move_smooth(SHOULDER_CH, sh_s)
    move_smooth(ELBOW_CH, el_s)

    gripper_close()

    try:
        requests.get("http://raspberrypi.local:5000/increment")
    except:
        pass

    time.sleep(1)
    go_home()

# ==========================================
# 🧠 YOLO
# ==========================================
interpreter = Interpreter(model_path="best_float16.tflite", num_threads=4)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

CONF = 0.6
RIPE_ID = 2

# ==========================================
# 🚀 START
# ==========================================
go_home()
time.sleep(3)

frame_count = 0

while True:

    with lock:
        if frame is None:
            continue
        img = frame.copy()

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    resized = cv2.resize(img, (inp[0]['shape'][2], inp[0]['shape'][1]))
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

            cx = int(x * img.shape[1])
            cy = int(y * img.shape[0])

            print("🍅 DETECTED")

            move_to_target(cx, cy, img)
            break
