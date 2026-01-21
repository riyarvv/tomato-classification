import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# =============================
# INITIALIZATION: ARM & PCA9685
# =============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

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

# =============================
# INITIALIZATION: TFLITE MODEL
# =============================
MODEL_PATH = "tomato_model_pi.tflite"  
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
HEALTHY_CLASS_INDEX = 0 

# =============================
# ARM CONTROL FUNCTIONS
# =============================
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
    print("🏠 Resetting to Neutral Position...")
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_and_drop():
    print("🍅 Ripe & Healthy detected! Starting Pick Sequence...")
    # Pick
    move_slow(BASE_CH, 40)
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], speed=0.02)
    time.sleep(1)
    servos[GRIPPER_CH].angle = None # Relax
    
    # Drop
    print("🧺 Moving to Drop Zone...")
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], speed=0.02)
    print("✅ Harvest Complete.")
    go_home()

# =============================
# MAIN VISION LOOP
# =============================
cap = cv2.VideoCapture(0)
go_home() # Ensure arm is out of camera way at start

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Red Detection
        mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 2000: continue # Ignore small dots
            
            x, y, w, h = cv2.boundingRect(cnt)
            tomato_crop = frame[y:y+h, x:x+w]
            if tomato_crop.size == 0: continue

            # Inference
            img = cv2.resize(tomato_crop, (224, 224)).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]

            # Logic: If Ripe (HSV passed) AND Healthy (TFLite passed)
            if class_idx == HEALTHY_CLASS_INDEX and confidence >= 0.60:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("Detection", frame)
                cv2.waitKey(1) # Refresh window
                
                pick_and_drop() # <--- Trigger the Arm
                break # Break contour loop to restart scan after picking

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pca.deinit()
