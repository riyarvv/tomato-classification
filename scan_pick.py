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
from flask import Flask, Response

# ==========================================
# 📏 ARM PARAMETERS
# ==========================================
L1, L2, L3 = 14.5, 13.5, 9.0
BASE_HEIGHT = 6.5
FOV_H, FOV_V = 48.8, 36.6

# ==========================================
# 🎯 SERVO OFFSETS (HOME = 20,160,20)
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
# 🎥 CAMERA THREAD
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
# 🏠 HOME
# ==========================================
def go_home():
    print("🏠 HOME (20,160,20)")
    move_smooth(BASE_CH, 20)
    move_smooth(SHOULDER_CH, 160)
    move_smooth(ELBOW_CH, 20)
    servos[GRIPPER_CH].angle = 100

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
    sh_s = SHOULDER_OFFSET + SHOULDER_DIR * shoulder
    el_s = ELBOW_OFFSET + ELBOW_DIR * elbow

    base_s = int(max(BASE_MIN, min(BASE_MAX, base_s)))
    sh_s = int(max(SH_MIN, min(SH_MAX, sh_s)))
    el_s = int(max(EL_MIN, min(EL_MAX, el_s)))

    return base_s, sh_s, el_s

# ==========================================
# 🎯 MOVE TO TARGET
# ==========================================
def move_to_target(cx, cy, img):

    Z = get_distance()
    X, Y = pixel_to_world(cx, cy, img.shape[1], img.shape[0], Z)

    angles = inverse_kinematics(X, Y, Z)
    if angles is None:
        print("❌ Out of reach")
        return

    base, sh, el = angles
    b, s, e = to_servo_angles(base, sh, el)

    print("🎯 Moving:", b, s, e)

    move_smooth(BASE_CH, b)
    move_smooth(SHOULDER_CH, s)
    move_smooth(ELBOW_CH, e)

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
# 🔄 SCANNING
# ==========================================
scan_angle = 20
scan_dir = 1
locked = False

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

    # 🔄 SCAN
    if not locked:
        scan_angle += scan_dir * 1
        if scan_angle >= 100 or scan_angle <= 20:
            scan_dir *= -1
        move_smooth(BASE_CH, scan_angle, delay=0.01)

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

            h_img, w_img = img.shape[:2]

            cx = int(x * w_img)
            cy = int(y * h_img)

            xmin = int((x - w/2) * w_img)
            ymin = int((y - h/2) * h_img)
            xmax = int((x + w/2) * w_img)
            ymax = int((y + h/2) * h_img)

            # 📦 DRAW
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
            cv2.circle(img, (cx,cy), 5, (0,0,255), -1)
            cv2.putText(img, f"Ripe {conf:.2f}",
                        (xmin,ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

            if not locked:
                print("🍅 TARGET LOCKED")

                locked = True

                with lock:
                    frame = img.copy()

                move_to_target(cx, cy, img)

                time.sleep(2)
                locked = False

            break

    # update stream frame
    with lock:
        frame = img.copy()
