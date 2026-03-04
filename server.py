from flask import Flask, Response
import serial
import cv2
import threading

app = Flask(__name__)

# ===== SERIAL CONNECTION =====
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# ===== CAMERA =====
camera = cv2.VideoCapture(0)

# ===== GLOBAL VARIABLES =====
harvesting = False
tomato_count = 0


# ================================
# SERIAL LISTENER THREAD (IMPORTANT)
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
# CAMERA ROUTE
# ================================
@app.route('/camera')
def camera_feed():
    ret, frame = camera.read()
    if not ret:
        return "Camera Error", 500

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


# ================================
# COUNT ROUTE
# ================================
@app.route('/count')
def count():
    return str(tomato_count)


# ================================
# PICK ROUTE (ONLY ARM)
# ================================
@app.route('/pick')
def pick():
    # arm_pick()
    return "Arm Activated"


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
    global harvesting
    harvesting = False
    ser.write(b'S')
    return "Harvest Stopped"


# ================================
# MANUAL MOVEMENT
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

    # Start serial listener thread
    serial_thread = threading.Thread(target=read_serial)
    serial_thread.daemon = True
    serial_thread.start()

    app.run(host="192.168.7.65", port=5000)
