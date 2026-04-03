from flask import Flask, Response, redirect, jsonify
import serial
import threading
import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# ================= GLOBAL =================
harvesting = False
tomato_count = 0

# ================= MODEL =================
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

input_h = inp[0]['shape'][1]
input_w = inp[0]['shape'][2]

CONF = 0.4
RIPE_ID = 2

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= ROBOT SETTINGS =================
BASE_MIN, BASE_MAX = 5, 160
FRAME_CENTER = 305
BASE_HOME = 20

base_angle = BASE_HOME
scan_dir = 1
centered = False

POSES = {
    "FAR": (110, 15, 70),
    "MID": (125, 10, 65),
    "CLOSE": (140, 6, 60)
}

HOME = (135, 5, 60)
current_pose = HOME

GRIP_CLOSE_SEQ = [100, 90, 75, 60, 45, 30, 15]
GRIP_OPEN_SEQ = [30, 45, 60, 75, 90, 100]

# ================= SERIAL FUNCTIONS =================
def send_base(a):
    ser.write(f"B,{a}\n".encode())

def send_pose(s, e, p):
    ser.write(f"A,{s},{e},{p}\n".encode())

def send_grip(g):
    ser.write(f"G,{g}\n".encode())

# ================= SERIAL LISTENER =================
def read_serial():
    global tomato_count

    while True:
        if ser.in_waiting:
            line = ser.readline().decode().strip()

            if line.startswith("COUNT:"):
                tomato_count = int(line.split(":")[1])
                print("Updated Count:", tomato_count)

# ================= PICK HELPERS =================
def smooth_move(start, end, steps=8, delay=0.08):
    s1, e1, p1 = start
    s2, e2, p2 = end

    for i in range(steps + 1):
        s = int(s1 + (s2 - s1) * i / steps)
        e = int(e1 + (e2 - e1) * i / steps)
        p = int(p1 + (p2 - p1) * i / steps)

        send_pose(s, e, p)
        time.sleep(delay)

def get_distance():
    if ser.in_waiting:
        try:
            return float(ser.readline().decode().strip())
        except:
            return None
    return None

def get_smart_pick_pose(dist, cy, frame_h):
    if dist > 25:
        s, e, p = POSES["FAR"]
    elif dist > 18:
        s, e, p = POSES["MID"]
    else:
        s, e, p = POSES["CLOSE"]

    img_center = frame_h // 2
    y_error = cy - img_center

    s -= int(y_error * 0.05)
    s += 5
    s = max(95, min(150, s))

    return (s, e, p)

# ================= VIDEO STREAM =================
def generate_frames():
    global base_angle, scan_dir, centered, current_pose, harvesting

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]

        # -------- TFLITE --------
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(inp[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(out[0]['index'])[0].T

        boxes, scores, centers = [], [], []

        for pred in output:
            x, y, bw, bh = pred[:4]
            scores_arr = pred[4:]

            cid = int(np.argmax(scores_arr))
            conf = scores_arr[cid]

            if conf > CONF and cid == RIPE_ID:
                cx = int(x * w)
                cy = int(y * h)

                xmin = int((x - bw/2) * w)
                ymin = int((y - bh/2) * h)
                xmax = int((x + bw/2) * w)
                ymax = int((y + bh/2) * h)

                boxes.append((xmin, ymin, xmax, ymax))
                scores.append(conf)
                centers.append((cx, cy))

        if len(boxes) > 0:
            best = int(np.argmax(scores))
            x1, y1, x2, y2 = boxes[best]
            cx, cy = centers[best]

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame, (cx,cy),5,(0,255,0),-1)

            error = cx - FRAME_CENTER

            if abs(error) > 30:
                base_angle -= int(error * 0.02)
                base_angle = max(BASE_MIN, min(BASE_MAX, base_angle))
                send_base(base_angle)
            else:
                centered = True

            if centered and harvesting:
                dist = get_distance()

                if dist:
                    target = get_smart_pick_pose(dist, cy, h)

                    smooth_move(current_pose, target)

                    for g in GRIP_CLOSE_SEQ:
                        send_grip(g)
                        time.sleep(0.2)

                    smooth_move(target, HOME)

                    for g in GRIP_OPEN_SEQ:
                        send_grip(g)
                        time.sleep(0.2)

                    centered = False

                    # update count
                    global tomato_count
                    tomato_count += 1
                    ser.write(f"COUNT:{tomato_count}\n".encode())

        else:
            base_angle += scan_dir * 2
            if base_angle >= BASE_MAX or base_angle <= BASE_MIN:
                scan_dir *= -1

            send_base(base_angle)

        # -------- STREAM --------
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================= ROUTES =================
@app.route("/control")
def control():

    return """
<html>

<head>
<title>Agribot Controller</title>

<style>
body{
font-family:Arial;
text-align:center;
background:#f4f4f4;
}

button{
width:140px;
height:70px;
font-size:18px;
margin:10px;
border-radius:10px;
border:none;
background:#4CAF50;
color:white;
}

button:hover{
background:#45a049;
}

.video{
border:5px solid black;
margin-top:20px;
}

.countbox{
font-size:28px;
margin-top:20px;
color:#333;
}
</style>

</head>

<body>

<h1>🤖 Agribot Controller</h1>

<h2>Live Camera</h2>

<img src="/video_feed" width="480" class="video">

<br><br>

<h2>Robot Movement</h2>

<button onclick="send('F')">⬆ Forward</button><br>

<button onclick="send('L')">⬅ Left</button>
<button onclick="send('S')">Stop</button>
<button onclick="send('R')">➡ Right</button><br>

<button onclick="send('B')">⬇ Back</button>

<br><br>

<h2>Motor Speed</h2>

<input type="range" min="0" max="255" value="180"
onchange="speed(this.value)">

<br><br>

<h2>Harvesting Control</h2>

<button onclick="startHarvest()">Start Harvest</button>
<button onclick="stopHarvest()">Stop Harvest</button>

<div class="countbox">
🍅 Tomato Count: <span id="count">0</span>
</div>

<br>

<button onclick="resetCount()">Reset Count</button>

<script>

function send(cmd){
fetch('/move/' + cmd)
}

function speed(val){
fetch('/speed/' + val)
}

function startHarvest(){
fetch('/pick')
}

function stopHarvest(){
fetch('/stop_harvest')
}

function resetCount(){
fetch('/reset')
}

function updateCount(){
fetch('/count')
.then(res => res.json())
.then(data => {
document.getElementById("count").innerText = data.count
})
}

setInterval(updateCount,1000)

</script>

</body>
</html>
"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pick')
def pick():
    global harvesting
    harvesting = True
    return "Harvesting Started"

@app.route('/stop_harvest')
def stop():
    global harvesting
    harvesting = False
    return "Harvesting Stopped"

@app.route('/count')
def get_count():
    return jsonify({"count": tomato_count})

# ================= MAIN =================
if __name__ == "__main__":

    threading.Thread(target=read_serial, daemon=True).start()

    print("🚀 Agribot Server Running")

    app.run(host="0.0.0.0", port=5000, threaded=True)
