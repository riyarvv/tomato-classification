import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. ARM INITIALIZATION (Added to your script)
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Map your working channels
BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH = 0, 1, 2, 3, 5

servos = {}
channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH]
for ch in channels:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

LIMITS = {
    BASE_CH:     {"neutral": 20,  "min": 10,  "max": 50},
    PITCH_CH:    {"neutral": 90,  "min": 40,  "max": 120},
    SHOULDER_CH: {"neutral": 130, "pick": 115},
    ELBOW_CH:    {"neutral": 65,  "pick": 100},
    GRIPPER_CH:  {"open": 170,    "close": 20}
}

def move_slow(channel_id, target_angle, speed=0.04):
    current = servos[channel_id].angle
    if current is None: current = 90
    start_angle, target_angle = int(current), int(target_angle)
    target_angle = max(0, min(180, target_angle))
    if start_angle == target_angle: return
    step = 1 if target_angle > start_angle else -1
    for angle in range(start_angle, target_angle + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

def go_home():
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_and_drop():
    # Sequence based on your requirements
    move_slow(BASE_CH, 40)
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], speed=0.02)
    time.sleep(1)
    servos[GRIPPER_CH].angle = None # Relax gripper
    # Return to neutral and release
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], speed=0.02)
    go_home()

# ==========================================
# 2. YOUR WORKING TFLITE LOGIC
# ==========================================
# NEW CODE (Paste this)
MODEL_PATH = "tomato_model_pi_v11.tflite"

try:
    from tflite_runtime.interpreter import Interpreter
    # Adding num_threads can sometimes help with initialization stability on Pi 5
    interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
    print("✅ Interpreter initialized with multi-threading")
except Exception as e:
    print(f"❌ Standard load failed: {e}")
    # Fallback to basic initialization
    interpreter = Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

# These must come AFTER allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
HEALTHY_CLASS_INDEX = 1

cap = cv2.VideoCapture(0)
go_home()

# ... (Keep your previous imports and initializations) ...

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. GET SCREEN DIMENSIONS & CALCULATE CENTER
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        
        # Define a "Target Zone" (e.g., a 100x100 pixel box in the center)
        zone_size = 100 
        zone_left = center_x - (zone_size // 2)
        zone_right = center_x + (zone_size // 2)
        zone_top = center_y - (zone_size // 2)
        zone_bottom = center_y + (zone_size // 2)

        # 2. DRAW CENTER CROSSHAIR (Visual Guide)
        # Vertical line
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        # Horizontal line
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        # Target Zone Box (Optional)
        cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (255, 255, 255), 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # CALCULATE THE CENTER OF THE TOMATO
            tomato_center_x = x + (w // 2)
            tomato_center_y = y + (h // 2)

            # AI Inference
            tomato_crop = frame[y:y+h, x:x+w]
            if tomato_crop.size == 0: continue
            img = cv2.resize(tomato_crop, (224, 224)).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

            # CHECK IF CENTERED: Is the tomato center inside our Target Zone?
            is_centered = (zone_left < tomato_center_x < zone_right) and \
                          (zone_top < tomato_center_y < zone_bottom)

            if class_idx == HEALTHY_CLASS_INDEX and confidence >= 0.60:
                label = "Healthy"
                color = (0, 255, 0)
                
                # ONLY TRIGGER ARM IF CENTERED
                if is_centered:
                    cv2.putText(frame, "TARGET LOCKED", (center_x - 50, center_y - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Harvest Vision", frame)
                    cv2.waitKey(1)
                    pick_and_drop()
                    break
                else:
                    cv2.putText(frame, "ALIGNING...", (x, y - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                label = "Unhealthy"
                color = (0, 0, 255)

            # Draw Labels
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.circle(frame, (tomato_center_x, tomato_center_y), 5, color, -1) # Tomato center dot
            cv2.putText(frame, f"Ripe {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Harvest Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
