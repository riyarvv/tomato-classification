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

app = Flask(__name__)

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

servos = {}
for ch in [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH]:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# ==========================================
# 3️⃣ SERVO LIMITS
# ==========================================
LIMITS = {
    BASE_CH:     {"min":10, "max":100},
    SHOULDER_CH: {"neutral":125, "pick":115},
    ELBOW_CH:    {"neutral":30,  "pick":50},
    PITCH_CH:    {"neutral":90},
    GRIPPER_CH:  {"close":15, "open":100}
}

CART_POSITION = 20

# ==========================================
# 4️⃣ SMOOTH MOVEMENT FUNCTION
# ==========================================
def move_smooth(channel, target, step=1, delay=0.03):

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
# 5️⃣ GRIPPER STEP MOVEMENT
# ==========================================
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

# ==========================================
# 6️⃣ INITIAL POSITION
# ==========================================
move_smooth(BASE_CH, 20)
move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
move_smooth(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]
servos[CAMERA_CH].angle = servos[BASE_CH].angle

# ==========================================
# 7️⃣ LOAD YOLO MODEL
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
# 8️⃣ CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

# shared frame for streaming
output_frame = None
lock = threading.Lock()

# ==========================================
# VIDEO STREAM GENERATOR
# ==========================================
def generate_frames():

    global output_frame

    while True:

        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# START FLASK SERVER
# ==========================================
def run_server():
    app.run(host="0.0.0.0", port=5001, threaded=True)

server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# ==========================================
# SCANNING VARIABLES
# ==========================================
scan_angle = 20
scan_direction = 1
locked = False
prev_time = 0

# ==========================================
# PICK FUNCTION
# ==========================================
def pick_and_drop():
    global scan_angle

    print("🍅 Picking Ripe Tomato...")

    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])

    gripper_close_slow()
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
    frame_count = 0
    while True:

        if not locked:

            scan_angle += scan_direction * 1

            if scan_angle >= LIMITS[BASE_CH]["max"] or scan_angle <= LIMITS[BASE_CH]["min"]:
                scan_direction *= -1

            move_smooth(BASE_CH, scan_angle, step=1, delay=0.02)
            servos[CAMERA_CH].angle = servos[BASE_CH].angle
            time.sleep(0.05)

        ret, frame = cap.read()
        if not ret:
            break
    
        frame_count += 1
    
        # Skip frames to speed up detection
        if frame_count % 4 != 0:
            with lock:
                output_frame = frame.copy()
            continue

        orig_h, orig_w = frame.shape[:2]
        center_x, center_y = orig_w//2, orig_h//2

        zone_size = 120
        zone_left = center_x - zone_size//2
        zone_right = center_x + zone_size//2
        zone_top = center_y - zone_size//2
        zone_bottom = center_y + zone_size//2

        cv2.rectangle(frame,(zone_left,zone_top),
                      (zone_right,zone_bottom),
                      (255,255,255),1)

        img = cv2.resize(frame,(input_w,input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        img = np.expand_dims(img,axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        output = output.T

        boxes = []
        scores = []
        centers = []

        for pred in output:
            x,y,w,h = pred[:4]
            class_scores = pred[4:]

            class_id = int(np.argmax(class_scores))
            confidence = class_scores[class_id]

            if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

                xmin = int((x - w/2) * orig_w)
                ymin = int((y - h/2) * orig_h)
                xmax = int((x + w/2) * orig_w)
                ymax = int((y + h/2) * orig_h)

                boxes.append([xmin,ymin,xmax-xmin,ymax-ymin])
                scores.append(float(confidence))
                centers.append((int(x*orig_w), int(y*orig_h)))

        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                   CONF_THRESHOLD, IOU_THRESHOLD)

        if len(indices) > 0:
            for idx in indices.flatten():

                x,y,bw,bh = boxes[idx]
                score = scores[idx]
                cx,cy = centers[idx]

                is_centered = (
                    zone_left < cx < zone_right and
                    zone_top < cy < zone_bottom
                )

                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
                cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
                cv2.putText(frame,f"Ripe {score:.2f}",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

                if is_centered and not locked:

                    print(f"🎯 Target locked at angle {scan_angle}")
                    locked = True
                    pick_and_drop()
                    time.sleep(2)
                    locked = False
                    break

        curr_time = time.time()
        fps = 1/(curr_time-prev_time+1e-5)
        prev_time = curr_time

        cv2.putText(frame,f"FPS: {int(fps)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),2)

        with lock:
            output_frame = frame.copy()

finally:
    cap.release()
    pca.deinit()
