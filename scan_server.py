from flask import Flask, Response
import serial
import cv2
import threading
import numpy as np
import subprocess
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# ================================
# SERIAL CONNECTION
# ================================
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# ================================
# CAMERA
# ================================
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================================
# YOLO TFLITE MODEL
# ================================
MODEL_PATH = "best_float16.tflite"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2
scan_process=None

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# ================================
# GLOBAL VARIABLES
# ================================
harvesting = False
tomato_count = 0

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

@app.route('/reset')
def reset():
    global tomato_count
    tomato_count = 0
    ser.write(b'Z')
    print("System Reset to 0")
    return {"status": "reset"}

# ================================
# YOLO LIVE VIDEO STREAM
# ================================
def generate_frames():

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        output = output.T

        boxes = []
        scores = []

        for pred in output:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]

            class_id = int(np.argmax(class_scores))
            confidence = class_scores[class_id]

            if confidence > CONF_THRESHOLD and class_id == RIPE_CLASS_ID:

                xmin = int((x - w/2) * orig_w)
                ymin = int((y - h/2) * orig_h)
                xmax = int((x + w/2) * orig_w)
                ymax = int((y + h/2) * orig_h)

                boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                scores.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                score = scores[i]

                label = f"Ripe {score:.2f}"

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0,255,0), 2)
                cv2.putText(frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================
# BASIC ROUTES
# ================================
@app.route("/")
def home():
    return "Agribot Server Running"

@app.route('/pick')
def pick():
    global scan_process

    if scan_process is None:
        print("Starting scan_pick.py...")
        scan_process = subprocess.Popen(
            ["python3", "scan_pick.py"]
        )
        return "Harvesting Started"
    else:
        return "Already Running"

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
    return "Stopped"

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
