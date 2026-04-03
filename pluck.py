import cv2
import serial
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

# ================= MODEL (TFLITE) =================
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# ================= BASE SETTINGS =================
BASE_MIN, BASE_MAX = 5, 160
FRAME_CENTER = 305
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

HOME = (135, 5, 60)
current_pose = HOME

# ================= GRIPPER =================
GRIP_CLOSE_SEQ = [100, 90, 75, 60, 45, 30, 15]
GRIP_OPEN_SEQ = [30, 45, 60, 75, 90, 100]

CONF = 0.4
RIPE_CLASS_ID = 2

# ================= SERIAL COMMANDS =================
def send_base(a):
    ser.write(f"B,{a}\n".encode())

def send_pose(s, e, p):
    ser.write(f"A,{s},{e},{p}\n".encode())

def send_grip(g):
    ser.write(f"G,{g}\n".encode())
    print("Gripper:", g)

# ================= SLOW SMOOTH MOVE =================
def smooth_move(start, end, steps=8, delay=0.08):
    s1, e1, p1 = start
    s2, e2, p2 = end

    for i in range(steps + 1):
        s = int(s1 + (s2 - s1) * i / steps)
        e = int(e1 + (e2 - e1) * i / steps)
        p = int(p1 + (p2 - p1) * i / steps)

        send_pose(s, e, p)
        time.sleep(delay)

# ================= SMART PICK HEIGHT =================
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
    s = max(95, min(150, s))

    return (s, e, p)

# ================= ULTRASONIC =================
def get_distance():
    if ser.in_waiting:
        try:
            return float(ser.readline().decode().strip())
        except:
            return None
    return None

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera failed ❌")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # ================= TFLITE INFERENCE =================
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.T

    found = False

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

            boxes.append((xmin, ymin, xmax, ymax))
            scores.append(confidence)
            centers.append((cx, cy))

    if len(boxes) > 0:
        best_idx = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[best_idx]
        cx, cy = centers[best_idx]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

        error = cx - FRAME_CENTER

        # ================= CENTER BASE =================
        if abs(error) > 30:
            base_angle -= int(error * 0.02)
            base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))
            send_base(base_angle)
            time.sleep(0.05)

        else:
            print("CENTERED ✅")
            centered = True
            time.sleep(0.2)

        # ================= PICK =================
        if centered:
            dist = get_distance()

            if dist is not None:
                target = get_smart_pick_pose(
                    dist, cy, frame.shape[0]
                )

                smooth_move(current_pose, target)

                for g in GRIP_CLOSE_SEQ:
                    send_grip(g)
                    time.sleep(0.2)

                print("GRIPPED 🍅")

                detach = (
                    max(95, target[0] - 2),
                    target[1],
                    target[2]
                )

                smooth_move(target, detach)

                smooth_move(detach, HOME)

                # base return
                if base_angle > BASE_HOME:
                    for angle in range(base_angle, BASE_HOME, -2):
                        send_base(angle)
                        time.sleep(0.08)
                else:
                    for angle in range(base_angle, BASE_HOME, 2):
                        send_base(angle)
                        time.sleep(0.08)

                base_angle = BASE_HOME

                for g in GRIP_OPEN_SEQ:
                    send_grip(g)
                    time.sleep(0.2)

                print("DROPPED IN CART ✅")

                centered = False
                time.sleep(1)

        found = True

    else:
        # ================= SCAN =================
        base_angle += scan_dir * 2

        if base_angle >= BASE_MAX or base_angle <= BASE_MIN:
            scan_dir *= -1

        base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))
        send_base(base_angle)
        time.sleep(0.08)

    cv2.imshow("Tomato Robot FINAL SLOW", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
