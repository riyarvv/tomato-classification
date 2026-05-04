import serial
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. INITIALIZE HARDWARE 
# ==========================================
arduino_port = '/dev/ttyUSB0'   # CHANGE back to /dev/ttyACM0 if needed!
camera_index = 0

print("Connecting to Arduino...")
try:
    arduino = serial.Serial(port=arduino_port, baudrate=9600, timeout=0.1)
    time.sleep(3)
    arduino.reset_input_buffer()
    print("Connection to Arduino successful!")
except Exception as e:
    print(f"Error: Could not connect to Arduino. {e}")
    exit()

print("Initializing Camera...")
cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ==========================================
# 2. LOAD TFLITE MODEL
# ==========================================
print("Loading TFLite model...")
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = input_details[0]['shape'][1]

print("Model loaded successfully!")

# ==========================================
# 3. CALIBRATION CONSTANTS
# ==========================================
PIXELS_PER_CM = 25.5
CAMERA_X_CM = -19.5
CAMERA_Y_CM = 15.2

# ==========================================
# 4. COORDINATE CONVERSION
# ==========================================
def convert_pixel_coords_to_robot_coords(pixel_x, pixel_y, frame_width, frame_height):
    # Tracking the TRUE center of the tomato
    raw_cam_x = (pixel_x - (frame_width / 2)) / PIXELS_PER_CM
    raw_cam_y = ((frame_height / 2) - pixel_y) / PIXELS_PER_CM

    # 1. Base offset
    robot_x = raw_cam_x + CAMERA_X_CM
    robot_y = raw_cam_y + CAMERA_Y_CM

    # 2. Physical Shoulder Compensation
    SHOULDER_COMPENSATION = 1.5
    robot_x += SHOULDER_COMPENSATION

    # 3. Nathan's Parallax Math (The Geometry Fix)
    CAMERA_HEIGHT_CM = 10.0
    TOMATO_RADIUS = 3.0
    parallax_error_y = TOMATO_RADIUS * (raw_cam_y / CAMERA_HEIGHT_CM)
    robot_y -= parallax_error_y

    # 4. Gripper Forward Reach Compensation
    FORWARD_REACH = 5.5
    robot_y += FORWARD_REACH

    # 5. Safe Grab Height
    robot_z = 5.0

    return round(robot_x, 1), round(robot_y, 1), round(robot_z, 1)

# ==========================================
# 5. PREPROCESS
# ==========================================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# ==========================================
# 6. POSTPROCESS (YOLOv8 + NMS Filter)
# ==========================================
def postprocess(output, frame_w, frame_h, conf_threshold=0.5, iou_threshold=0.4):
    # This function now perfectly mimics the Ultralytics filter
    output = np.squeeze(output).T

    boxes = []
    scores = []
    class_ids = []
    centers = []

    for det in output:
        x_center, y_center, w, h = det[:4]
        class_probs = det[4:] 
        
        class_id = np.argmax(class_probs)
        conf = class_probs[class_id]

        if conf < conf_threshold:
            continue

        x_center = x_center * frame_w
        y_center = y_center * frame_h
        width = w * frame_w
        height = h * frame_h

        x_min = int(x_center - (width / 2))
        y_min = int(y_center - (height / 2))

        boxes.append([x_min, y_min, int(width), int(height)])
        scores.append(float(conf))
        class_ids.append(int(class_id))
        centers.append((x_center, y_center))

    detections = []
    
    # Apply Non-Maximum Suppression (NMS) to delete overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                "x": centers[i][0],  
                "y": centers[i][1],
                "x_min": boxes[i][0], 
                "y_min": boxes[i][1],
                "w": boxes[i][2],
                "h": boxes[i][3],
                "conf": scores[i],
                "class": class_ids[i]
            })

    return detections

# ==========================================
# 7. MAIN LOOP
# ==========================================
print("Starting main loop... Press 'q' to exit.")

last_pick_time = 0

while cap.isOpened():

    # Arduino messages (Terminal is clean now, so you will actually see these!)
    while arduino.in_waiting > 0:
        try:
            msg = arduino.readline().decode('utf-8').strip()
            if msg:
                print(f"[ARDUINO SAYS]: {msg}")
        except:
            pass

    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # -------- INFERENCE --------
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    detections = postprocess(output_data, frame_w, frame_h)

    current_time = time.time()

    for det in detections:
        class_id = det["class"]
        conf = det["conf"]
        
        # --- DRAWING THE BOXES ON THE SCREEN ---
        cv2.rectangle(frame, (det["x_min"], det["y_min"]), 
                     (det["x_min"] + det["w"], det["y_min"] + det["h"]), 
                     (0, 255, 0), 2)
        
        label = f"Class {class_id}: {conf:.2f}"
        cv2.putText(frame, label, (det["x_min"], det["y_min"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 🔥 TRIGGER LOGIC (Class 2 is Ripe)
        if class_id == 2 and (current_time - last_pick_time > 25):

            px_x = det["x"]
            px_y = det["y"]

            rx, ry, rz = convert_pixel_coords_to_robot_coords(px_x, px_y, frame_w, frame_h)

            # SAFETY CHECK
            if ry > 26.0 or ry < 5.0:
                print(f"\n[SAFETY STOP] Target Y:{ry} is out of safe physical reach! Ignoring.")
                last_pick_time = time.time() - 20
                continue

            command = f"{rx},{ry},{rz}\n"
            print(f"\n>>> TFLite TRIGGERED! Sending: {command}")

            arduino.write(command.encode())
            arduino.flush()

            last_pick_time = time.time()
            print(">>> Waiting 25 seconds for the Arduino to finish its sequence...\n")

    cv2.imshow("Tomato Detection (TFLite Pi)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
