from flask import Flask, Response, jsonify
import serial
import threading
import subprocess
import cv2

app = Flask(__name__)

# ================================
# SERIAL CONNECTIONS
# ================================
esp = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# ================================
# CAMERA INIT (STREAM INSIDE SERVER)
# ================================
cap = cv2.VideoCapture(0)

# ================================
# GLOBAL VARIABLES
# ================================
harvesting = False
tomato_count = 0
scan_process = None

# ================================
# SERIAL LISTENER (ESP COUNT)
# ================================
def read_serial():
    global tomato_count

    while True:
        if esp.in_waiting:
            line = esp.readline().decode().strip()

            if line.startswith("COUNT:"):
                tomato_count = int(line.split(":")[1])
                print("🍅 Updated Count:", tomato_count)

# ================================
# VIDEO STREAM GENERATOR
# ================================
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # (optional) resize for speed
        frame = cv2.resize(frame, (480, 320))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ================================
# CONTROL PAGE
# ================================
@app.route("/control")
def control():
    return """
<html>
<head>
<title>Agribot Controller</title>
<style>
body{font-family:Arial;text-align:center;background:#f4f4f4;}
button{width:140px;height:70px;font-size:18px;margin:10px;border-radius:10px;border:none;background:#4CAF50;color:white;}
button:hover{background:#45a049;}
.video{border:5px solid black;margin-top:20px;}
.countbox{font-size:28px;margin-top:20px;color:#333;}
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
function send(cmd){ fetch('/move/' + cmd) }
function speed(val){ fetch('/speed/' + val) }
function startHarvest(){ fetch('/pick') }
function stopHarvest(){ fetch('/stop_harvest') }
function resetCount(){ fetch('/reset') }

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

# ================================
# VIDEO FEED (NOW LOCAL)
# ================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================
# COUNT ROUTES
# ================================
@app.route("/count")
def get_count():
    return jsonify({"count": tomato_count})

@app.route("/increment")
def increment():
    global tomato_count

    tomato_count += 1
    esp.write(f"COUNT:{tomato_count}\n".encode())

    print("🍅 Tomato Count:", tomato_count)

    return {"count": tomato_count}

@app.route("/reset")
def reset():
    global tomato_count
    tomato_count = 0
    esp.write(b'Z\n')
    return {"status": "reset"}

# ================================
# START HARVESTING
# ================================
@app.route('/pick')
def pick():
    global scan_process, harvesting

    if scan_process is None:
        scan_process = subprocess.Popen(
            ["python3", "pluck.py"]
        )
        harvesting = True
        print("🌱 Harvesting started")
        return "Harvesting Started"

    return "Already Running"

# ================================
# STOP HARVESTING
# ================================
@app.route('/stop_harvest')
def stop_harvest():
    global scan_process, harvesting

    harvesting = False

    if scan_process:
        scan_process.terminate()
        scan_process = None

    return "Stopped"

# ================================
# ROBOT MOVEMENT (ESP)
# ================================
@app.route('/move/<cmd>')
def move(cmd):
    cmd = cmd.upper()

    if cmd in ['F','B','L','R','S']:
        esp.write((cmd + '\n').encode())
        return f"Sent {cmd}"

    return "Invalid"

# ================================
# SPEED CONTROL (ESP)
# ================================
@app.route('/speed/<int:value>')
def speed(value):
    value = max(0, min(255, value))
    esp.write(f"SPEED:{value}\n".encode())
    return {"speed": value}

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    threading.Thread(target=read_serial, daemon=True).start()

    print("🚀 Server Running at http://<PI-IP>:5001/control")

    app.run(host="0.0.0.0", port=5001, threaded=True)
