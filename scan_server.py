from flask import Flask
import serial
import threading
import subprocess

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
# COUNT ROUTES
# ================================
@app.route("/count")
def get_count():
    return {"count": tomato_count}

@app.route("/reset")
def reset():
    global tomato_count
    tomato_count = 0
    ser.write(b'Z')  # Reset command to ESP32
    print("System Reset to 0")
    return {"status": "reset"}

# ================================
# HOME
# ================================
@app.route("/")
def home():
    return "Agribot Server Running"

# ================================
# START HARVESTING (RUN scan_pick.py)
# ================================
@app.route('/pick')
def pick():

    global scan_process

    if scan_process is None or scan_process.poll() is not None:

        print("Starting scan_pick.py ...")

        scan_process = subprocess.Popen(
            ["python3", "scan_pick.py"]
        )

        return "Harvesting Started"

    else:
        return "Already Running"

# ================================
# START / STOP
# ================================
@app.route('/start')
def start():
    global harvesting
    harvesting = True
    return "Harvest Started"

@app.route('/stop')
def stop():

    global harvesting, scan_process

    harvesting = False

    ser.write(b'S')

    if scan_process is not None:
        scan_process.terminate()
        scan_process = None
        print("Harvesting Stopped")

    return "Harvest Stopped"

# ================================
# ROBOT MOVEMENT
# ================================
@app.route('/forward')
def forward():
    ser.write(b'F')
    return "Forward"

@app.route('/back')
def back():
    ser.write(b'B')
    return "Back"

@app.route('/left')
def left():
    ser.write(b'L')
    return "Left"

@app.route('/right')
def right():
    ser.write(b'R')
    return "Right"

# ================================
# MAIN
# ================================
if __name__ == "__main__":

    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True
    serial_thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)
