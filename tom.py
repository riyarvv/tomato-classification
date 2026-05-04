import serial
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. INITIALIZE HARDWARE 
# ==========================================
arduino_port = '/dev/ttyUSB0'   # CHANGE if needed
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
    raw_cam_x = (pixel_x - (frame_width / 2)) / PIXELS_PER_CM
    raw_cam_y = ((frame_height / 2) - pixel_y) / PIXELS_PER_CM

    robot_x = raw_cam_x + CAMERA_X_CM
    robot_y = raw_cam_y + CAMERA_Y_CM

    SHOULDER_COMPENSATION = 1.5
    robot_x += SHOULDER_COMPENSATION

    CAMERA_HEIGHT_CM = 10.0
    TOMATO_RADIUS = 3.0
    parallax_error_y = TOMATO_RADIUS * (raw_cam_y / CAMERA_HEIGHT_CM)
    robot_y -= parallax_error_y

    FORWARD_REACH = 5.5
    robot_y += FORWARD_REACH

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
# 6. POSTPROCESS (YOLOv8 TFLite Fix)
# ==========================================
def postprocess(output, frame_w, frame_h, conf_threshold=0.5):
    detections = []
    
    # Remove batch and transpose: (1, 7, 8400) -> (8400, 7)
    output = np.squeeze(output).T

    for det in output:
        # Indices 0, 1, 2, 3 are x, y, w, h
        x_center, y_center, w, h = det[:4]
        
        # Indices 4, 5, 6 are the actual class probabilities
        class_probs = det[4:] 
        
        class_id = np.argmax(class_probs)
        conf = class_probs[class_id]

        if conf < conf_threshold:
            continue

        # Convert normalized coordinates back to actual pixels
        x_center = x_center * frame_w
        y_center = y_center * frame_h
        width = w * frame_w
        height = h * frame_h

        # Calculate top-left corner for drawing boxes later
        x_min = int(x_center - (width / 2))
        y_min = int(y_center - (height / 2))

        detections.append({
            "x": x_center,  # Keep center for robot math
            "y": y_center,
            "x_min": x_min, # Keep top-left for drawing
            "y_min": y_min,
            "w": int(width),
            "h": int(height),
            "conf": float(conf),
            "class": int(class_id)
        })

    return detections
# ==========================================
# 7. MAIN LOOP
# ==========================================
print("Starting main loop... Press 'q' to exit.")

last_pick_time = 0

while cap.isOpened():

    # Arduino messages
    while arduino.in_waiting > 0:
        try:
            msg = arduino.readline().decode('utf-8').strip()
            if msg:
                print(f"[ARDUINO]: {msg}")
        except:
            pass

    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # -------- INFERENCE --------
    input_data = preprocess(frame)
    print("thinking")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    print("done thinking")
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Output shape: {output_data.shape}")

    detections = postprocess(output_data, frame_w, frame_h)

    current_time = time.time()

    for det in detections:
        class_id = det["class"]
        conf = det["conf"]
        
        # --- DRAWING THE BOXES ON THE SCREEN ---
        # Draw a green rectangle around the detection
        cv2.rectangle(frame, (det["x_min"], det["y_min"]), 
                     (det["x_min"] + det["w"], det["y_min"] + det["h"]), 
                     (0, 255, 0), 2)
        
        # Put the class ID and confidence above the box
        label = f"Class {class_id}: {conf:.2f}"
        cv2.putText(frame, label, (det["x_min"], det["y_min"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # ---------------------------------------

        # 🔥 Adjust class ID if needed (Assuming 0 is "Ripe")
        if class_id == 0 and (current_time - last_pick_time > 25):

            px_x = det["x"]
            px_y = det["y"]

            rx, ry, rz = convert_pixel_coords_to_robot_coords(px_x, px_y, frame_w, frame_h)

            # SAFETY CHECK
            if ry > 26.0 or ry < 5.0:
                print(f"\n[SAFETY STOP] Y:{ry} out of range")
                last_pick_time = time.time() - 20
                continue

            command = f"{rx},{ry},{rz}\n"
            print(f"\n>>> TFLite TRIGGERED! Sending: {command}")

            arduino.write(command.encode())
            arduino.flush()

            last_pick_time = time.time()
            print(">>> Waiting 25 seconds...\n")
    cv2.imshow("Tomato Detection (TFLite Pi)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
