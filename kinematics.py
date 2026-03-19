#!/usr/bin/env python3
"""
Smart Tomato Harvesting Robot with Inverse Kinematics and Ultrasonic Sensor
Complete solution for professional tomato harvesting
"""

import time
import board
import busio
import cv2
import numpy as np
import RPi.GPIO as GPIO
import threading
import math
from collections import deque
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter
from flask import Flask, Response, jsonify
import requests

# ==========================================
# FLASK APP INITIALIZATION
# ==========================================
app = Flask(__name__)

# ==========================================
# ULTRASONIC SENSOR CLASS
# ==========================================
class UltrasonicSensor:
    """
    Measures distance to tomato for Z-axis (depth)
    Uses HC-SR04 ultrasonic sensor
    """
    
    def __init__(self, trig_pin=23, echo_pin=24):
        self.TRIG = trig_pin
        self.ECHO = echo_pin
        self.current_distance = 0
        self.running = True
        
        # For smooth readings
        self.distance_history = deque(maxlen=5)
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        
        # Start continuous reading thread
        self.thread = threading.Thread(target=self._continuous_reading)
        self.thread.daemon = True
        self.thread.start()
        print("✅ Ultrasonic sensor initialized")
        
    def _measure_distance(self):
        """Single distance measurement"""
        # Send trigger pulse
        GPIO.output(self.TRIG, False)
        time.sleep(0.000002)
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)
        
        # Wait for echo start with timeout
        timeout = time.time() + 0.1
        pulse_start = time.time()
        while GPIO.input(self.ECHO) == 0:
            if time.time() > timeout:
                return None
            pulse_start = time.time()
        
        # Wait for echo end
        while GPIO.input(self.ECHO) == 1:
            if time.time() > timeout:
                return None
            pulse_end = time.time()
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound formula
        return round(distance, 2)
    
    def _continuous_reading(self):
        """Background thread for continuous readings"""
        while self.running:
            dist = self._measure_distance()
            if dist and 2 < dist < 400:  # Valid range
                self.distance_history.append(dist)
                # Use median to filter noise
                if len(self.distance_history) > 2:
                    sorted_dists = sorted(self.distance_history)
                    self.current_distance = sorted_dists[len(sorted_dists)//2]
            time.sleep(0.05)
    
    def get_distance(self):
        """Get current filtered distance"""
        return self.current_distance
    
    def cleanup(self):
        self.running = False
        time.sleep(0.2)
        GPIO.cleanup()

# ==========================================
# INVERSE KINEMATICS CLASS
# ==========================================
class InverseKinematics:
    """
    Converts (X, Y, Z) coordinates to servo angles
    X = left/right (from camera)
    Y = height (from camera)
    Z = depth (from ultrasonic)
    """
    
    def __init__(self):
        # ========== MEASURE YOUR ARM LENGTHS ==========
        # Adjust these values to match your actual robot arm (in cm)
        self.L1 = 14.5.0  # Shoulder to elbow
        self.L2 = 13.5.0  # Elbow to wrist
        self.L3 = 9.0   # Wrist to gripper tip
        
        # Base offset from ground (cm)
        self.base_height = 6.50
        
        # Servo angle limits
        self.base_min = 10
        self.base_max = 100
        self.shoulder_min = 0
        self.shoulder_max = 180
        self.elbow_min = 0
        self.elbow_max = 180
        
        # Camera parameters
        self.FOV_HORIZONTAL = 62  # degrees
        self.FOV_VERTICAL = 48    # degrees
        
    def pixel_to_cm(self, pixel_x, pixel_y, distance_z, frame_width, frame_height):
        """
        Convert camera pixels to real-world centimeters
        """
        if distance_z <= 0:
            return 0, 0, 0
            
        # Calculate real-world width at distance Z
        real_width_at_z = 2 * distance_z * math.tan(math.radians(self.FOV_HORIZONTAL/2))
        real_height_at_z = 2 * distance_z * math.tan(math.radians(self.FOV_VERTICAL/2))
        
        # Pixels to cm ratio
        cm_per_pixel_x = real_width_at_z / frame_width
        cm_per_pixel_y = real_height_at_z / frame_height
        
        # Convert from image center
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Calculate offset from center in cm
        offset_x_cm = (pixel_x - center_x) * cm_per_pixel_x
        offset_y_cm = (center_y - pixel_y) * cm_per_pixel_y  # Y increases downward in image
        
        # Final position
        x_cm = offset_x_cm
        y_cm = self.base_height + offset_y_cm  # Height from ground
        z_cm = distance_z  # Forward distance
        
        return x_cm, y_cm, z_cm
    
    def calculate_angles(self, x, y, z):
        """
        Calculate servo angles to reach point (x, y, z)
        Returns: (base_angle, shoulder_angle, elbow_angle) in degrees
        """
        try:
            # Step 1: Calculate base angle (rotation)
            base_angle = math.degrees(math.atan2(x, z))
            
            # Step 2: Calculate horizontal distance from base
            horizontal_dist = math.sqrt(x**2 + z**2)
            
            # Step 3: Calculate distance from shoulder to target
            dx = horizontal_dist
            dy = y - self.base_height
            
            # Distance from shoulder to target
            D = math.sqrt(dx**2 + dy**2)
            
            # Check if reachable
            max_reach = self.L1 + self.L2 + self.L3
            if D > max_reach:
                print(f"⚠️ Target too far: {D:.1f}cm > max {max_reach:.1f}cm")
                return None
            
            # Step 4: Calculate elbow angle using law of cosines
            cos_elbow = (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)
            cos_elbow = max(-1, min(1, cos_elbow))  # Clamp to valid range
            elbow_angle = math.degrees(math.acos(cos_elbow))
            
            # Step 5: Calculate shoulder angle
            phi1 = math.atan2(dy, dx)
            phi2 = math.acos((self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D))
            shoulder_angle = math.degrees(phi1 - phi2)
            
            # Step 6: Adjust for servo mounting orientation
            base_final = base_angle
            shoulder_final = 180 - (shoulder_angle + 90)
            elbow_final = 180 - elbow_angle
            
            # Apply limits
            base_final = max(self.base_min, min(self.base_max, base_final))
            shoulder_final = max(self.shoulder_min, min(self.shoulder_max, shoulder_final))
            elbow_final = max(self.elbow_min, min(self.elbow_max, elbow_final))
            
            return {
                'base': base_final,
                'shoulder': shoulder_final,
                'elbow': elbow_final
            }
        except Exception as e:
            print(f"Error in IK calculation: {e}")
            return None
    
    def calculate_gripper_angle(self, y, z):
        """
        Calculate wrist pitch angle to keep gripper level
        """
        try:
            pitch_angle = math.degrees(math.atan2(y - self.base_height, z))
            return max(0, min(180, 90 - pitch_angle))
        except:
            return 90

# ==========================================
# PCA9685 INITIALIZATION
# ==========================================
print("Initializing PCA9685...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==========================================
# CHANNEL MAPPING
# ==========================================
BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH = 0, 1, 2, 6, 5, 3

servos = {}
for ch in [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH, CAMERA_CH]:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# ==========================================
# SERVO LIMITS
# ==========================================
LIMITS = {
    BASE_CH:     {"min": 10, "max": 100},
    SHOULDER_CH: {"neutral": 160, "pick": 140},
    ELBOW_CH:    {"neutral": 20,  "pick": 30},
    PITCH_CH:    {"neutral": 90},
    GRIPPER_CH:  {"close": 15, "open": 100}
}

CART_POSITION = 20

# ==========================================
# SMOOTH MOVEMENT FUNCTION
# ==========================================
def move_smooth(channel, target, step=1, delay=0.02):
    """Move servo smoothly to target angle"""
    current = servos[channel].angle
    if current is None:
        current = target

    current = int(current)
    target = int(target)

    if current < target:
        angles = range(current, target + 1, step)
    else:
        angles = range(current, target - 1, -step)

    for angle in angles:
        servos[channel].angle = angle
        time.sleep(delay)

# ==========================================
# GRIPPER FUNCTIONS
# ==========================================
def gripper_open_slow():
    """Open gripper slowly"""
    steps = [15, 30, 45, 60, 75, 90, 100]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(0.3)

def gripper_close_slow():
    """Close gripper slowly"""
    steps = [100, 90, 75, 60, 45, 30, 15]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(0.3)

def gripper_open_fast():
    """Open gripper quickly"""
    servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]

def gripper_close_fast():
    """Close gripper quickly"""
    servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["close"]

# ==========================================
# HOME POSITION FUNCTION
# ==========================================
def return_home():
    """Return arm to neutral position"""
    print("🏠 Returning to home position...")
    move_smooth(BASE_CH, 20, step=2, delay=0.02)
    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"], step=2, delay=0.02)
    move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"], step=2, delay=0.02)
    move_smooth(PITCH_CH, LIMITS[PITCH_CH]["neutral"], step=2, delay=0.02)
    servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]
    servos[CAMERA_CH].angle = servos[BASE_CH].angle

# ==========================================
# INITIAL POSITION
# ==========================================
print("Moving to initial position...")
move_smooth(BASE_CH, 20)
move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
move_smooth(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["open"]
servos[CAMERA_CH].angle = servos[BASE_CH].angle
print("✅ Initial position set")

# ==========================================
# INITIALIZE SENSORS
# ==========================================
ultrasonic = UltrasonicSensor()
ik = InverseKinematics()
print("✅ Sensors initialized")

# ==========================================
# LOAD YOLO MODEL
# ==========================================
MODEL_PATH = "/home/rslvpi5/tomato-detection/tomato-classification/best_float16.tflite"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
RIPE_CLASS_ID = 2

print("Loading YOLO model...")
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]
print("✅ Model loaded")

# ==========================================
# CAMERA SETUP
# ==========================================
print("Initializing camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Shared frame for streaming
output_frame = None
lock = threading.Lock()
print("✅ Camera initialized")

# ==========================================
# TARGET TRACKING
# ==========================================
target_tomato = {
    'pixel_x': None,
    'pixel_y': None,
    'distance': None,
    'x_cm': None,
    'y_cm': None,
    'z_cm': None,
    'angles': None,
    'confidence': 0
}

# ==========================================
# VIDEO STREAM GENERATOR
# ==========================================
def generate_frames():
    global output_frame

    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode('.jpg', output_frame, 
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """API endpoint for robot status"""
    return jsonify({
        'state': 'scanning' if not locked else 'picking',
        'tomatoes_found': len([t for t in [target_tomato] if t['pixel_x']]),
        'battery': 100,  # Add battery monitoring if available
        'distance': ultrasonic.get_distance()
    })

@app.route('/api/command/<cmd>', methods=['POST'])
def api_command(cmd):
    """API endpoint for commands"""
    global locked, scan_angle, scan_direction
    
    if cmd == 'home':
        return_home()
        return jsonify({'message': 'Returning home'})
    elif cmd == 'scan':
        locked = False
        scan_direction = 1
        return jsonify({'message': 'Scanning started'})
    elif cmd == 'stop':
        locked = True
        return jsonify({'message': 'Stopped'})
    elif cmd == 'emergency':
        emergency_stop()
        return jsonify({'message': 'EMERGENCY STOP'})
    else:
        return jsonify({'message': 'Unknown command'})

# ==========================================
# START FLASK SERVER
# ==========================================
def run_server():
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=False)

server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()
print("✅ Flask server started on port 5001")

# ==========================================
# SCANNING VARIABLES
# ==========================================
scan_angle = 20
scan_direction = 1
locked = False
prev_time = 0
frame_count = 0

# ==========================================
# AUTO ALIGN WITH 3D
# ==========================================
def auto_align_3d(cx, cy, frame_width, frame_height):
    """
    3D alignment using X (camera), Y (camera), Z (ultrasonic)
    """
    global scan_angle, target_tomato
    
    # Get distance from ultrasonic
    distance_z = ultrasonic.get_distance()
    
    if distance_z < 5 or distance_z > 50:
        print(f"⚠️ Bad distance: {distance_z}cm")
        return False
    
    # Convert pixels to real-world coordinates
    x_cm, y_cm, z_cm = ik.pixel_to_cm(cx, cy, distance_z, 
                                      frame_width, frame_height)
    
    # Calculate required servo angles
    angles = ik.calculate_angles(x_cm, y_cm, z_cm)
    
    if angles is None:
        print("❌ Target unreachable")
        return False
    
    # Store target information
    target_tomato = {
        'pixel_x': cx,
        'pixel_y': cy,
        'distance': distance_z,
        'x_cm': x_cm,
        'y_cm': y_cm,
        'z_cm': z_cm,
        'angles': angles,
        'confidence': 1.0
    }
    
    # Check if we're aligned (small error tolerance)
    current_base = servos[BASE_CH].angle
    if current_base is None:
        current_base = scan_angle
        
    base_error = abs(angles['base'] - current_base)
    
    if base_error < 2:  # Within 2 degrees
        print(f"✅ Aligned! Target at {x_cm:.1f}cm, {y_cm:.1f}cm, {z_cm:.1f}cm")
        return True
    else:
        # Move towards target
        if angles['base'] > current_base:
            scan_angle += 1
        else:
            scan_angle -= 1
        
        scan_angle = max(LIMITS[BASE_CH]["min"],
                        min(LIMITS[BASE_CH]["max"], scan_angle))
        
        move_smooth(BASE_CH, scan_angle, step=1, delay=0.01)
        servos[CAMERA_CH].angle = servos[BASE_CH].angle
        return False

# ==========================================
# PICK WITH INVERSE KINEMATICS
# ==========================================
def pick_with_ik():
    """
    Pick tomato using calculated IK angles
    """
    global target_tomato, locked
    
    if not target_tomato or 'angles' not in target_tomato or target_tomato['angles'] is None:
        print("❌ No target data")
        locked = False
        return
    
    print(f"\n🍅 Picking tomato at:")
    print(f"   Position: X={target_tomato['x_cm']:.1f}cm, "
          f"Y={target_tomato['y_cm']:.1f}cm, Z={target_tomato['z_cm']:.1f}cm")
    
    angles = target_tomato['angles']
    
    # PHASE 1: PRE-PICK POSITION (10cm before and above tomato)
    print("📍 Phase 1/7: Moving to pre-pick position...")
    
    # Calculate pre-pick position
    pre_x = target_tomato['x_cm']
    pre_y = target_tomato['y_cm'] + 5  # 5cm above
    pre_z = target_tomato['z_cm'] - 10  # 10cm behind
    
    pre_angles = ik.calculate_angles(pre_x, pre_y, pre_z)
    
    if pre_angles:
        # Move to pre-pick position
        move_smooth(BASE_CH, pre_angles['base'], step=2, delay=0.02)
        move_smooth(SHOULDER_CH, pre_angles['shoulder'], step=2, delay=0.02)
        move_smooth(ELBOW_CH, pre_angles['elbow'], step=2, delay=0.02)
        
        # Calculate and set wrist pitch
        pitch = ik.calculate_gripper_angle(target_tomato['y_cm'], target_tomato['z_cm'])
        move_smooth(PITCH_CH, pitch, step=2, delay=0.02)
        
        # Open gripper
        gripper_open_slow()
        time.sleep(1)
    
    # PHASE 2: APPROACH TOMATO
    print("📍 Phase 2/7: Approaching tomato...")
    
    # Move to exact tomato position
    move_smooth(BASE_CH, angles['base'], step=1, delay=0.01)
    move_smooth(SHOULDER_CH, angles['shoulder'], step=1, delay=0.01)
    move_smooth(ELBOW_CH, angles['elbow'], step=1, delay=0.01)
    
    # Final distance check
    final_dist = ultrasonic.get_distance()
    if abs(final_dist - target_tomato['distance']) > 5:
        print("⚠️ Tomato moved! Aborting...")
        return_home()
        locked = False
        return
    
    time.sleep(1)
    
    # PHASE 3: GRASP
    print("📍 Phase 3/7: Grasping tomato...")
    gripper_close_slow()
    time.sleep(1)
    
    # Check if we have something
    # You could add a pressure sensor or current monitoring here
    
    # PHASE 4: DETACH (gentle twist)
    print("📍 Phase 4/7: Detaching from plant...")
    
    # Small upward movement
    current_shoulder = servos[SHOULDER_CH].angle
    move_smooth(SHOULDER_CH, current_shoulder + 5, step=1, delay=0.02)
    time.sleep(0.3)
    
    # Gentle twist
    current_elbow = servos[ELBOW_CH].angle
    move_smooth(ELBOW_CH, current_elbow + 5, step=1, delay=0.02)
    time.sleep(0.3)
    move_smooth(ELBOW_CH, current_elbow - 5, step=1, delay=0.02)
    time.sleep(0.3)
    
    # PHASE 5: RETRACT
    print("📍 Phase 5/7: Retracting from plant...")
    
    # Move up and back
    retract_x = target_tomato['x_cm']
    retract_y = target_tomato['y_cm'] + 15  # 15cm up
    retract_z = target_tomato['z_cm'] - 20  # 20cm back
    
    retract_angles = ik.calculate_angles(retract_x, retract_y, retract_z)
    
    if retract_angles:
        move_smooth(ELBOW_CH, retract_angles['elbow'], step=2, delay=0.02)
        move_smooth(SHOULDER_CH, retract_angles['shoulder'], step=2, delay=0.02)
    
    # PHASE 6: MOVE TO CART
    print("📍 Phase 6/7: Moving to collection cart...")
    move_smooth(BASE_CH, CART_POSITION, step=2, delay=0.02)
    
    # PHASE 7: DROP
    print("📍 Phase 7/7: Dropping tomato...")
    gripper_open_slow()
    
    # PHASE 8: RETURN HOME
    return_home()
    
    print("✅ Pick complete! Tomato harvested successfully.\n")
    locked = False

# ==========================================
# EMERGENCY STOP
# ==========================================
def emergency_stop():
    """Emergency stop - freeze all movement"""
    global locked
    print("🚨 EMERGENCY STOP ACTIVATED!")
    locked = True
    # Don't move servos, just stop

# ==========================================
# CALIBRATION FUNCTION
# ==========================================
def calibrate_system():
    """
    Calibrate camera-to-robot coordinate system
    """
    print("\n🔧 Starting calibration...")
    print("Place a known object at these positions and verify arm reaches it.")
    
    test_positions = [
        (0, 15, 20),   # (x cm, y cm, z cm) - center
        (10, 15, 25),  # right
        (-10, 20, 20), # left, higher
    ]
    
    for i, (tx, ty, tz) in enumerate(test_positions):
        print(f"\nTest {i+1}: Move object to X={tx}cm, Y={ty}cm, Z={tz}cm")
        input("Press Enter when ready...")
        
        # Calculate expected angles
        angles = ik.calculate_angles(tx, ty, tz)
        if angles:
            print(f"Calculated angles: Base={angles['base']:.1f}°, "
                  f"Shoulder={angles['shoulder']:.1f}°, Elbow={angles['elbow']:.1f}°")
            
            # Move to position
            move_smooth(BASE_CH, angles['base'])
            move_smooth(SHOULDER_CH, angles['shoulder'])
            move_smooth(ELBOW_CH, angles['elbow'])
            
            response = input("Is the arm pointing at the object? (y/n): ")
            if response.lower() != 'y':
                print("Adjust IK parameters in the InverseKinematics class")
        else:
            print("Position unreachable")
    
    return_home()
    print("✅ Calibration complete!")

# ==========================================
# MAIN LOOP
# ==========================================
try:
    print("\n🚀 Smart Harvester Robot Started!")
    print("📡 Web interface available at: http://raspberrypi.local:5001")
    print("🔍 Scanning for tomatoes...\n")
    
    while True:
        # SCANNING MOVEMENT (if not locked)
        if not locked:
            scan_angle += scan_direction * 1

            if scan_angle >= LIMITS[BASE_CH]["max"] or scan_angle <= LIMITS[BASE_CH]["min"]:
                scan_direction *= -1

            move_smooth(BASE_CH, scan_angle, step=1, delay=0.02)
            servos[CAMERA_CH].angle = servos[BASE_CH].angle
            time.sleep(0.05)

        # CAPTURE FRAME
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames to speed up detection
        if frame_count % 4 != 0:
            with lock:
                output_frame = frame.copy()
            continue

        # FRAME DIMENSIONS
        orig_h, orig_w = frame.shape[:2]
        center_x = orig_w // 2
        center_y = int(orig_h * 0.75)   # 75% down (bottom half center)

        # DRAW TARGET ZONE
        zone_size = 120
        zone_left = center_x - zone_size//2
        zone_right = center_x + zone_size//2
        zone_top = center_y - zone_size//2
        zone_bottom = center_y + zone_size//2

        cv2.rectangle(frame, (zone_left, zone_top),
                      (zone_right, zone_bottom),
                      (255, 255, 255), 1)

        # PREPARE IMAGE FOR MODEL
        img = cv2.resize(frame, (input_w, input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # RUN INFERENCE
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        output = output.T

        # PROCESS DETECTIONS
        boxes = []
        scores = []
        centers = []

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
                centers.append((int(x * orig_w), int(y * orig_h)))

        # APPLY NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                score = scores[idx]
                cx, cy = centers[idx]

                # Check if in zone
                in_zone = (zone_left < cx < zone_right and 
                          zone_top < cy < zone_bottom)

                # Draw detection
                color = (0, 255, 0) if in_zone else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                
                label = f"Ripe {score:.2f}"
                if in_zone:
                    label += " ✓"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ATTEMPT PICK if in zone and not locked
                if not locked and in_zone:
                    print(f"🎯 Tomato detected in target zone! Confidence: {score:.2f}")
                    
                    # Try to align with 3D
                    aligned = auto_align_3d(cx, cy, orig_w, orig_h)
                    
                    if aligned:
                        print(f"🎯 Target acquired! Starting harvest sequence...")
                        locked = True
                        pick_with_ik()
                        time.sleep(2)
                        locked = False
                        break

        # CALCULATE AND DISPLAY FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        # ADD INFO TO FRAME
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        status = "LOCKED" if locked else "SCANNING"
        cv2.putText(frame, f"Status: {status}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 0, 255) if locked else (0, 255, 0), 2)
        
        # Show distance
        dist = ultrasonic.get_distance()
        cv2.putText(frame, f"Distance: {dist:.1f}cm", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # UPDATE OUTPUT FRAME
        with lock:
            output_frame = frame.copy()

except KeyboardInterrupt:
    print("\n\n🛑 Program interrupted by user")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🧹 Cleaning up...")
    return_home()
    time.sleep(1)
    cap.release()
    pca.deinit()
    ultrasonic.cleanup()
    print("✅ Cleanup complete. Goodbye!") 
