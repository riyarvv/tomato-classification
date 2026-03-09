from flask import Flask, Response, redirect, jsonify
import serial
import threading
import subprocess
import cv2

app = Flask(__name__)

# ================================
# SERIAL CONNECTION (ESP32)
# ================================
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# ================================
# GLOBAL VARIABLES
# ================================
harvesting = False
tomato_count = 0
scan_process = None

# ================================
# SERIAL LISTENER THREAD
# ================================
def read_serial():
    global tomato_count

    while True:

        if ser.in_waiting:
            line = ser.readline().decode().strip()

            if line.startswith("COUNT:"):
                tomato_count = int(line.split(":")[1])
                print("Updated Count:", tomato_count)

# ================================
# MAIN CONTROL PAGE
# ================================
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

# ================================
# VIDEO STREAM
# ================================
@app.route('/video_feed')
def video_feed():
    return redirect("http://raspberrypi.local:5001/video_feed")

# ================================
# COUNT ROUTES
# ================================
@app.route("/count")
def get_count():
    return jsonify({"count": tomato_count})

@app.route("/reset")
def reset():
    global tomato_count

    tomato_count = 0
    ser.write(b'Z')

    print("System Reset to 0")

    return {"status": "reset"}

# ================================
# START HARVESTING
# ================================
@app.route('/pick')
def pick():

    global scan_process, harvesting

    if scan_process is None or scan_process.poll() is not None:

        print("Starting scan_pick.py ...")
        scan_process = subprocess.Popen([
"/home/rslvpi5/tomato-detection/tomato-classification/venv/bin/python",
"scan_pick.py"
])
        
        harvesting = True

        return "Harvesting Started"

    return "Already Running"

# ================================
# STOP HARVESTING
# ================================
@app.route('/stop_harvest')
def stop_harvest():

    global scan_process, harvesting

    harvesting = False

    if scan_process is not None:
        scan_process.terminate()
        scan_process = None

        print("Harvesting Stopped")

    return "Harvesting Stopped"

# ================================
# ROBOT MOVEMENT
# ================================
@app.route('/move/<cmd>')
def move(cmd):

    global scan_process, harvesting

    cmd = cmd.upper()

    if cmd == "F":
        ser.write(b'F')
        return "Forward"

    elif cmd == "B":
        ser.write(b'B')
        return "Back"

    elif cmd == "L":
        ser.write(b'L')
        return "Left"

    elif cmd == "R":
        ser.write(b'R')
        return "Right"

    elif cmd == "S":

        harvesting = False
        ser.write(b'S')

        if scan_process is not None:
            scan_process.terminate()
            scan_process = None

        return "Stop"

    return "Invalid Command"

# ================================
# MOTOR SPEED CONTROL
# ================================
@app.route('/speed/<int:value>')
def set_speed(value):

    value = max(0, min(255, value))

    command = f"SPEED:{value}\n"
    ser.write(command.encode())

    print("Speed Set To:", value)

    return {"speed": value}

# ================================
# HOME
# ================================
@app.route("/")
def home():
    return "Agribot Server Running"

# ================================
# MAIN
# ================================
if __name__ == "__main__":

    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True
    serial_thread.start()

    print("Agribot Server Running...")

    app.run(host="0.0.0.0", port=5000, threaded=True)
