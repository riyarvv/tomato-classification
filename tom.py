import serial
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. INITIALIZE HARDWARE 
# ==========================================
arduino_port = '/dev/ttyUSB0'   # Adjust for Windows (COMX) or Linux (/dev/ttyUSB0)
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
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# 2. LOAD TFLITE MODEL
# ==========================================
print("Loading TFLite model...")
interpreter = Interpreter(model_path="best_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = input_details[0]['shape'][1] # Usually 640 or 320

# ==========================================
# 3. CALIBRATION & COORDINATES (Kept from your working code)
# ==========================================
PIXELS_PER_CM = 25.5
CAMERA_X_CM = -19.5
CAMERA_Y_CM = 15.2

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
# 4. TFLITE HELPER FUNCTIONS
# ==========================================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output_data, frame_w, frame_h, conf_threshold=0.6):
    detections = []
    # Shape is (1, 7, 8400). Squeeze to (7, 8400) then Transpose to (8400, 7)
    output = np.squeeze(output_data).T 

    for det in output:
        # For a (8400, 7) output:
        # Indices 0-3 are x, y, w, h
        # Indices 4-6 are class scores (assuming 3 classes or 1 conf + 2 classes)
        scores = det[4:] 
        conf = np.max(scores)
        class_id = np.argmax(scores)

        if conf > conf_threshold:
            # YOLO TFLite usually outputs normalized center coordinates
            x_center = det[0] * frame_w
            y_center = det[1] * frame_h
            detections.append({
                "x": x_center, "y": y_center, 
                "conf": conf, "class": int(class_id)
            })
    return detections

# ==========================================
# 5. MAIN LOOP
# ==========================================
last_pick_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Arduino feedback
    if arduino.in_waiting > 0:
        msg = arduino.readline().decode('utf-8', errors='ignore').strip()
        if msg: print(f"[ARDUINO]: {msg}")

    # Inference
    input_tensor = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    detections = postprocess(output_data, frame.shape[1], frame.shape[0])

    current_time = time.time()

    for det in detections:
        # Assuming Class 0 is your "Ripe" tomato
        if det["class"] == 0 and (current_time - last_pick_time > 25):
            rx, ry, rz = convert_pixel_coords_to_robot_coords(det["x"], det["y"], 1280, 720)

            # Safety check
            if 5.0 <= ry <= 26.0:
                command = f"{rx},{ry},{rz}\n"
                print(f"\n>>> TARGET SPOTTED! Sending: {command}")
                arduino.write(command.encode())
                arduino.flush()
                last_pick_time = current_time
            else:
                print(f"Tomato out of reach (Y: {ry})")

        # Visual feedback on screen
        cv2.circle(frame, (int(det["x"]), int(det["y"])), 5, (0, 255, 0), -1)

    cv2.imshow('TFLite Tomato Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
