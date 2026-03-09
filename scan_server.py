from flask import Flask, Response
import serial
import threading
import subprocess
import cv2
from flask import redirect

app = Flask(__name__)

# ================================
# SERIAL CONNECTION (ESP32)
# ================================
ser = serial.Serial('/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A58043556-if00', 115200, timeout=1)

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
    ser.write(b'Z')

    print("System Reset to 0")

    return {"status": "reset"}


# ================================
# VIDEO STREAM
# ================================
@app.route('/video_feed')
def video_feed():
    return redirect("http://raspberrypi.local:5001/video_feed")

# ================================
# HOME
# ================================
@app.route("/")
def home():
    return "Agribot Server Running"


# ================================
# START HARVESTING
# ================================
@app.route('/pick')
def pick():

    global scan_process, harvesting

    if scan_process is None or scan_process.poll() is not None:

        print("Starting scan_pick.py ...")

        scan_process = subprocess.Popen(
            ["python3", "scan_pick.py"]
        )

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


@app.route('/stop')
def stop():

    global scan_process, harvesting

    harvesting = False

    ser.write(b'S')

    if scan_process is not None:
        scan_process.terminate()
        scan_process = None

    return "Robot Stopped"

# ================================
# ROBOT MOVEMENT (STREAMLIT COMPATIBLE)
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
# MAIN
# ================================
if __name__ == "__main__":

    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True
    serial_thread.start()

    print("Agribot Server Running...")

    app.run(host="0.0.0.0", port=5000, threaded=True)
