import cv2
import serial
import time
import requests
import numpy as np
from tflite_runtime.interpreter import Interpreter
from flask import Flask, Response
import threading

stream_app = Flask(__name__)

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

# ================= MODEL =================
interpreter = Interpreter(model_path="best_float16.tflite", num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# ================= BASE SETTINGS =================
BASE_MIN, BASE_MAX = 5, 160
BASE_HOME = 20

base_angle = BASE_HOME
scan_dir = 1
centered = False

# ================= ARM POSES =================
POSES = {
    "FAR": (110, 15, 70),
    "MID": (125, 10, 65),
    "CLOSE": (140, 6, 60)
}

HOME = (155, 5, 60)
current_pose = HOME

# ================= GRIPPER =================
GRIP_CLOSE_SEQ = [80, 75, 60, 45, 30, 15]
GRIP_OPEN_SEQ = [30, 45, 60, 75, 80]

CONF = 0.4
RIPE_CLASS_ID = 2

# ================= SERIAL COMMANDS =================
def send_base(a):
    ser.write(f"B,{a}\n".encode())

def send_pose(s, e, p):
    ser.write(f"A,{s},{e},{p}\n".encode())

def send_grip(g):
    ser.write(f"G,{g}\n".encode())

# ================= SMOOTH MOVE =================
def smooth_move(start, end, steps=8, delay=0.08):
    s1, e1, p1 = start
    s2, e2, p2 = end

    for i in range(steps + 1):
        s = int(s1 + (s2 - s1) * i / steps)
        e = int(e1 + (e2 - e1) * i / steps)
        p = int(p1 + (p2 - p1) * i / steps)

        send_pose(s, e, p)
        time.sleep(delay)

# ================= SMART PICK =================
def get_smart_pick_pose(dist, cy, frame_h):
    if dist > 25:
        s, e, p = POSES["FAR"]
    elif dist > 18:
        s, e, p = POSES["MID"]
    else:
        s, e, p = POSES["CLOSE"]

    img_center = frame_h // 2
    y_error = cy - img_center

    shoulder_offset = int(y_error * 0.05)
    s = s - shoulder_offset

    s += 5
    s = max(110, min(150, s))

    return (s, e, p)

# ================= DISTANCE =================
def get_distance():
    try:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            return float(line)
    except:
        return None
    return None

# ================= CAMERA (MATCH scan_pick) =================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera failed ❌")
    exit()

output_frame = None
lock = threading.Lock()

# ================= STREAM =================
def generate():
    global output_frame

    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode(
                '.jpg', output_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@stream_app.route('/video_feed')
def video_feed():
    return Response(generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream():
    stream_app.run(host='0.0.0.0', port=5002, threaded=True)

threading.Thread(target=run_stream, daemon=True).start()

# ================= MAIN LOOP =================
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 Skip more frames → BIG lag reduction
    if frame_count % 4 != 0:
        with lock:
            output_frame = frame.copy()
        continue

    orig_h, orig_w = frame.shape[:2]
    FRAME_CENTER = orig_w // 2

    # ================= INFERENCE =================
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0].T

    boxes = []
    scores = []
    centers = []

    for pred in output:
        x, y, w, h = pred[:4]
        class_scores = pred[4:]

        class_id = int(np.argmax(class_scores))
        confidence = class_scores[class_id]

        if confidence > CONF and class_id == RIPE_CLASS_ID:

            cx = int(x * orig_w)
            cy = int(y * orig_h)

            xmin = int((x - w/2) * orig_w)
            ymin = int((y - h/2) * orig_h)
            xmax = int((x + w/2) * orig_w)
            ymax = int((y + h/2) * orig_h)

            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            scores.append(float(confidence))
            centers.append((cx, cy))

    # ================= NMS (IMPORTANT) =================
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF, 0.45)

    valid_centers = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cx, cy = centers[i]
            conf = scores[i]

            valid_centers.append((cx, cy))

            # ✅ DRAW BOX
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

            # ✅ LABEL (Ripe + Confidence)
            label = f"Ripe {conf:.2f}"
            cv2.putText(frame, label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

    # ================= ROBOT CONTROL =================
    if len(valid_centers) > 0:
        cx, cy = valid_centers[0]
        error = cx - FRAME_CENTER

        if abs(error) > 30:
            base_angle -= int(error * 0.02)
            base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))
            send_base(base_angle)
        else:
            centered = True

        if centered:
            dist = get_distance()

            if dist is not None:
                target = get_smart_pick_pose(dist, cy, frame.shape[0])

                smooth_move(current_pose, target)

                for g in GRIP_CLOSE_SEQ:
                    send_grip(g)

                    # 🍅 COUNT TRIGGER
                    if g == 30:
                        try:
                            requests.get("http://localhost:5001/increment", timeout=0.2)
                        except:
                            pass

                    time.sleep(0.15)   # 🔥 reduced delay

                detach = (max(95, target[0]-2), target[1], target[2])

                smooth_move(target, detach)
                smooth_move(detach, HOME)

                # return base
                step = -2 if base_angle > BASE_HOME else 2
                for angle in range(base_angle, BASE_HOME, step):
                    send_base(angle)

                base_angle = BASE_HOME

                for g in GRIP_OPEN_SEQ:
                    send_grip(g)
                    time.sleep(0.15)

                centered = False
                time.sleep(0.5)

    else:
        base_angle += scan_dir * 2

        if base_angle >= BASE_MAX or base_angle <= BASE_MIN:
            scan_dir *= -1

        base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))
        send_base(base_angle)

    # ================= STREAM UPDATE =================
    with lock:
        output_frame = frame.copy()

    cv2.imshow("Pluck Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
