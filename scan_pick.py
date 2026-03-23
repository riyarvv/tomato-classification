import time
import board
import busio
import cv2
import numpy as np
import threading
import requests

from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from gpiozero import DistanceSensor
from flask import Flask, Response

print("🚀 scan_pick.py STARTED")

# ==========================================
# SERVO LIMITS + HOME
# ==========================================
BASE_MIN, BASE_MAX = 10, 100
SH_MIN, SH_MAX = 130, 170
EL_MIN, EL_MAX = 20, 65

HOME_BASE = 20
HOME_SH = 160
HOME_EL = 20

# ==========================================
# HARDWARE
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
# CAMERA THREAD
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame = None
lock = threading.Lock()

def cam_thread():
    global frame
    while True:
        ret, img = cap.read()
        if ret:
            with lock:
                frame = img.copy()

threading.Thread(target=cam_thread, daemon=True).start()

# ==========================================
# FLASK STREAM
# ==========================================
app = Flask(__name__)

def generate():
    global frame
    while True:
        with lock:
            if frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5001, threaded=True),
    daemon=True
).start()

# ==========================================
# UTIL FUNCTIONS
# ==========================================
def move_smooth(ch, target, delay=0.01):
    current = servos[ch].angle or target
    step = 1 if target > current else -1
    for a in range(int(current), int(target), step):
        servos[ch].angle = a
        time.sleep(delay)

def go_home():
    print("🏠 HOME")
    move_smooth(BASE_CH, HOME_BASE)
    move_smooth(SHOULDER_CH, HOME_SH)
    move_smooth(ELBOW_CH, HOME_EL)
    servos[GRIPPER_CH].angle = 100

def get_distance():
    vals = []
    for _ in range(5):
        vals.append(sensor.distance * 100)
        time.sleep(0.05)
    return sum(vals)/len(vals)

def gripper_close():
    for a in [100,80,60,40,20,15]:
        servos[GRIPPER_CH].angle = a
        time.sleep(0.2)

# ==========================================
# YOLO
# ==========================================
interpreter = Interpreter(model_path="best_float16.tflite", num_threads=4)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

CONF = 0.6
RIPE_ID = 2

# ==========================================
# TRACK + PICK (CORE FIX)
# ==========================================
def track_and_pick():

    print("🎯 Tracking...")

    while True:

        with lock:
            if frame is None:
                continue
            img = frame.copy()

        h, w = img.shape[:2]

        # YOLO
        resized = cv2.resize(img, (inp[0]['shape'][2], inp[0]['shape'][1]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)/255
        rgb = np.expand_dims(rgb, axis=0)

        interpreter.set_tensor(inp[0]['index'], rgb)
        interpreter.invoke()

        output = interpreter.get_tensor(out[0]['index'])[0].T

        found = False

        for pred in output:

            x,y,w_box,h_box = pred[:4]
            scores = pred[4:]

            cid = np.argmax(scores)
            conf = scores[cid]

            if conf > CONF and cid == RIPE_ID:

                found = True

                cx = int(x * w)
                cy = int(y * h)

                # DRAW BOX
                xmin = int((x - w_box/2) * w)
                ymin = int((y - h_box/2) * h)
                xmax = int((x + w_box/2) * w)
                ymax = int((y + h_box/2) * h)

                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                cv2.circle(img,(cx,cy),5,(0,255,0),-1)

                # ===== ALIGN =====
                center_x = w//2
                center_y = int(h*0.7)

                error_x = cx - center_x
                error_y = cy - center_y

                # BASE
                base = servos[BASE_CH].angle + int(error_x * 0.05)
                base = max(BASE_MIN, min(BASE_MAX, base))
                servos[BASE_CH].angle = base

                # SHOULDER
                shoulder = servos[SHOULDER_CH].angle - int(error_y * 0.03)
                shoulder = max(SH_MIN, min(SH_MAX, shoulder))
                servos[SHOULDER_CH].angle = shoulder

                # ===== PICK =====
                if abs(error_x) < 20 and abs(error_y) < 20:

                    print("✅ ALIGNED")

                    while True:

                        dist = get_distance()

                        if dist < 6:
                            print("📍 Reached")
                            break

                        current = servos[ELBOW_CH].angle

                        if current >= EL_MAX:
                            break

                        move_smooth(ELBOW_CH, current + 2, 0.05)

                    print("🤏 GRIP")
                    gripper_close()

                    try:
                        requests.get("http://raspberrypi.local:5000/increment")
                    except:
                        pass

                    go_home()
                    return

                break

        if not found:
            print("❌ Lost target")
            return

# ==========================================
# MAIN LOOP
# ==========================================
go_home()
time.sleep(2)

scan_angle = 20
direction = 1
locked = False

while True:

    with lock:
        if frame is None:
            continue
        img = frame.copy()

    # 🔄 SCAN
    if not locked:
        scan_angle += direction

        if scan_angle >= BASE_MAX or scan_angle <= BASE_MIN:
            direction *= -1

        servos[BASE_CH].angle = scan_angle
        time.sleep(0.02)

    # YOLO QUICK CHECK
    resized = cv2.resize(img, (inp[0]['shape'][2], inp[0]['shape'][1]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)/255
    rgb = np.expand_dims(rgb, axis=0)

    interpreter.set_tensor(inp[0]['index'], rgb)
    interpreter.invoke()

    output = interpreter.get_tensor(out[0]['index'])[0].T

    for pred in output:

        scores = pred[4:]
        cid = np.argmax(scores)
        conf = scores[cid]

        if conf > CONF and cid == RIPE_ID:

            print("🍅 DETECTED → LOCK")

            locked = True
            track_and_pick()
            locked = False

            break

    with lock:
        frame = img.copy()

    cv2.imshow("Detection", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pca.deinit()
