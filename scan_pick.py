import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

from flask import Flask, Response
import threading

# ✅ Ultrasonic
from gpiozero import DistanceSensor
sensor = DistanceSensor(echo=24, trigger=23)

app = Flask(__name__)

# ==========================================
# PCA9685 INITIALIZATION
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==========================================
# CHANNEL MAPPING
# ==========================================
BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH = 0,1,2,6,5,3

servos = {}
for ch in [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH]:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# ==========================================
# SERVO LIMITS
# ==========================================
LIMITS = {
    BASE_CH:     {"min":10, "max":100},
    SHOULDER_CH: {"neutral":160, "pick":140},
    ELBOW_CH:    {"neutral":20,  "pick":30},
    PITCH_CH:    {"neutral":90},
    GRIPPER_CH:  {"close":15, "open":100}
}

CART_POSITION = 20

# ==========================================
# SMOOTH MOVEMENT
# ==========================================
def move_smooth(channel, target, step=1, delay=0.3):
    current = servos[channel].angle
    if current is None:
        current = target

    current = int(current)
    target = int(target)

    if current < target:
        angles = range(current, target + 1, step)
    else:
        angles = range(current, target - 1, -step)

    for angle in angles:
        servos[channel].angle = angle
        time.sleep(delay)

# ==========================================
# GRIPPER
# ==========================================
def gripper_open_slow():
    for angle in [15,30,45,60,75,90,100]:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)

def gripper_close_slow():
    for angle in [100,90,75,60,45,30,15]:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)

# ==========================================
# ULTRASONIC FUNCTION
# ==========================================
def get_stable_distance():
    readings = []
    for _ in range(5):
        readings.append(sensor.distance * 100)
        time.sleep(0.05)
    return sum(readings)/len(readings)

# ==========================================
# INITIAL POSITION
# ==========================================
move_smooth(BASE_CH, 20)
move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
move_smooth(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]
servos[CAMERA_CH].angle = servos[BASE_CH].angle

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = "/home/rslvpi5/tomato-detection/tomato-classification/best_float16.tflite"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2

interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

output_frame = None
lock = threading.Lock()

# ==========================================
# STREAM
# ==========================================
def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

def run_server():
    app.run(host="0.0.0.0", port=5001, threaded=True)

threading.Thread(target=run_server, daemon=True).start()

# ==========================================
# ALIGN
# ==========================================
scan_angle = 20
scan_direction = 1
locked = False
lock_counter = 0

def auto_align(cx, center_x):
    global scan_angle

    error = cx - center_x
    if abs(error) < 20:
        return True

    scan_angle += 1 if error > 0 else -1
    scan_angle = max(10, min(100, scan_angle))

    move_smooth(BASE_CH, scan_angle)
    servos[CAMERA_CH].angle = servos[BASE_CH].angle

    return False

# ==========================================
# PICK FUNCTION (ULTRASONIC BASED)
# ==========================================
def pick_and_drop():
    global scan_angle

    print("🍅 Picking Ripe Tomato...")

    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])

    # 🔥 ultrasonic forward motion (KEEP THIS NEW)
    while True:

        dist = get_stable_distance()
        print(f"Distance: {dist:.2f} cm")

        if dist < 8:
            print("📍 Close enough to pick")
            break

        current = servos[ELBOW_CH].angle

        if current >= 50:
            print("⚠️ Max reach reached")
            break

        # ⚠️ IMPORTANT: keep movement smooth but not too slow
        move_smooth(ELBOW_CH, current + 1, step=1, delay=0.05)

    time.sleep(1)

    gripper_close_slow()
    time.sleep(1)

    # your original detach motion
    move_smooth(ELBOW_CH, 25, step=2, delay=0.01)

    time.sleep(1)

    move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])

    move_smooth(BASE_CH, CART_POSITION)
    servos[CAMERA_CH].angle = servos[BASE_CH].angle
    scan_angle = CART_POSITION

    gripper_open_slow()
    time.sleep(1)

    print("✅ Pick Complete")

# ==========================================
# MAIN LOOP
# ==========================================
try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center_x = w // 2

        # dummy detection (keep your original YOLO code here)

        cx = center_x  # replace with actual detection center
        cy = int(h * 0.7)

        if not locked:
            aligned = auto_align(cx, center_x)

            if aligned:
                lock_counter += 1
            else:
                lock_counter = 0

            if lock_counter > 5:
                locked = True
                pick_and_drop()
                locked = False
                lock_counter = 0

        with lock:
            output_frame = frame.copy()

finally:
    cap.release()
    pca.deinit()
