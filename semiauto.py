#!/usr/bin/env python3
"""
Semi-Automatic Tomato Harvesting Robot
- Detects ripe tomatoes using YOLO
- Auto-aligns to detected tomatoes
- Manual confirmation before picking
- Emergency stop always available
"""

import time
import board
import busio
import cv2
import numpy as np
import threading
import math
import json
from collections import deque
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from gpiozero import DistanceSensor, Device
from gpiozero.pins.lgpio import LGPIOFactory
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from tflite_runtime.interpreter import Interpreter

# Set pin factory for Raspberry Pi 5
Device.pin_factory = LGPIOFactory()

# ==========================================
# FLASK APP INITIALIZATION
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tomato-robot-semi-auto'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ==========================================
# SERVO CONTROLLER (Using YOUR tested angles)
# ==========================================
class ServoController:
    def __init__(self, pca, channel, name, min_angle, max_angle, 
                 default_angle, inverted=False):
        self.channel = channel
        self.name = name
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = default_angle
        self.inverted = inverted
        
        try:
            self.servo = servo.Servo(pca.channels[channel], 
                                    min_pulse=500, 
                                    max_pulse=2500)
            self.set_angle(default_angle, smooth=False)
            print(f"✅ Servo {name}: {min_angle}°-{max_angle}°, default {default_angle}°")
        except Exception as e:
            print(f"❌ Error initializing {name}: {e}")
            self.servo = None
    
    def set_angle(self, angle, smooth=True, step_delay=0.02):
        """Set servo angle with optional smooth movement"""
        # Apply inversion if needed
        if self.inverted:
            angle = 180 - angle
            
        # Clamp to safe limits (YOUR tested values)
        angle = max(self.min_angle, min(self.max_angle, angle))
        
        if smooth and self.servo and abs(angle - self.current_angle) > 5:
            # Smooth movement for larger moves
            start = self.current_angle
            steps = int(abs(angle - start) / 2)  # 2 degree steps
            if steps > 0:
                step = 2 if angle > start else -2
                for i in range(steps + 1):
                    new_angle = start + (step * i)
                    self.servo.angle = new_angle
                    time.sleep(step_delay)
            self.servo.angle = angle
        elif self.servo:
            self.servo.angle = angle
            
        self.current_angle = angle
    
    def get_angle(self):
        return self.current_angle

# ==========================================
# ROBOT ARM WITH YOUR TESTED ANGLES
# ==========================================
class RobotArm:
    def __init__(self):
        print("Initializing PCA9685...")
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        
        # YOUR TESTED SERVO CONFIGURATION
        self.servos = {
            'base': ServoController(self.pca, 0, 'Base', 
                                   min_angle=10, max_angle=100, 
                                   default_angle=55, inverted=False),
            'shoulder': ServoController(self.pca, 1, 'Shoulder',
                                       min_angle=30, max_angle=160,
                                       default_angle=90, inverted=False),
            'elbow': ServoController(self.pca, 2, 'Elbow',
                                    min_angle=20, max_angle=90,
                                    default_angle=45, inverted=False),
            'wrist': ServoController(self.pca, 3, 'Wrist',
                                    min_angle=0, max_angle=180,
                                    default_angle=90, inverted=False),
            'gripper': ServoController(self.pca, 5, 'Gripper',
                                      min_angle=15, max_angle=100,
                                      default_angle=100, inverted=False),  # 100 = open
            'camera': ServoController(self.pca, 6, 'Camera',
                                     min_angle=10, max_angle=100,
                                     default_angle=55, inverted=False)
        }
        
        # YOUR TESTED PRESET POSITIONS
        self.positions = {
            'home': {
                'base': 55,
                'shoulder': 90,
                'elbow': 45,
                'wrist': 90,
                'gripper': 100  # Open
            },
            'scan': {
                'base': 55,
                'shoulder': 90,
                'elbow': 45,
                'wrist': 90,
                'gripper': 100
            },
            'pickup': {
                'base': 55,
                'shoulder': 140,  # YOUR tested pickup angle
                'elbow': 30,      # YOUR tested pickup angle
                'wrist': 45,      # Tilt for better grip
                'gripper': 100    # Open initially
            },
            'grasp': {
                'gripper': 15     # YOUR tested closed position
            },
            'lift': {
                'shoulder': 110,  # Lift slightly
                'elbow': 40
            },
            'retract': {
                'shoulder': 90,   # Return to home height
                'elbow': 45
            },
            'release': {
                'base': 20,       # Turn to collection cart
                'shoulder': 90,
                'elbow': 45,
                'wrist': 90,
                'gripper': 100    # Open to drop
            }
        }
        
        # State tracking
        self.current_state = 'idle'
        self.emergency_stopped = False
        self.camera_follow = True
        self.auto_align = True
        
        print("✅ Robot arm initialized with tested angles")
    
    def move_to_position(self, position_name, smooth=True):
        """Move to a preset position"""
        if self.emergency_stopped:
            return False
            
        if position_name not in self.positions:
            return False
            
        print(f"📍 Moving to {position_name} position...")
        pos = self.positions[position_name]
        
        # Move in order: base first, then arm, then gripper last
        if 'base' in pos:
            self.servos['base'].set_angle(pos['base'], smooth)
            if self.camera_follow:
                self.servos['camera'].set_angle(pos['base'], smooth)
            time.sleep(0.3)
            
        if 'shoulder' in pos:
            self.servos['shoulder'].set_angle(pos['shoulder'], smooth)
            time.sleep(0.2)
            
        if 'elbow' in pos:
            self.servos['elbow'].set_angle(pos['elbow'], smooth)
            time.sleep(0.2)
            
        if 'wrist' in pos:
            self.servos['wrist'].set_angle(pos['wrist'], smooth)
            time.sleep(0.1)
            
        if 'gripper' in pos:
            self.servos['gripper'].set_angle(pos['gripper'], smooth)
            time.sleep(0.2)
        
        print(f"✅ At {position_name} position")
        return True
    
    def align_to_tomato(self, x_offset, y_offset):
        """Auto-align base and shoulder to tomato position"""
        if self.emergency_stopped or not self.auto_align:
            return False
        
        # Convert pixel offset to angle adjustment
        # Assumes 640x480 frame, 48° horizontal FOV
        base_angle = self.servos['base'].get_angle()
        shoulder_angle = self.servos['shoulder'].get_angle()
        
        # Adjust base (left/right)
        base_adjustment = (x_offset / 640) * 45  # 45° total range
        new_base = base_angle + base_adjustment
        new_base = max(10, min(100, new_base))
        
        # Adjust shoulder (up/down)
        shoulder_adjustment = (y_offset / 480) * 70  # 70° total range
        new_shoulder = shoulder_angle - shoulder_adjustment  # Negative because y increases down
        new_shoulder = max(30, min(160, new_shoulder))
        
        # Smooth movement to target
        if abs(new_base - base_angle) > 1:
            self.servos['base'].set_angle(new_base, smooth=True)
            if self.camera_follow:
                self.servos['camera'].set_angle(new_base, smooth=True)
                
        if abs(new_shoulder - shoulder_angle) > 1:
            self.servos['shoulder'].set_angle(new_shoulder, smooth=True)
        
        # Check if aligned (within tolerance)
        aligned = (abs(new_base - base_angle) < 2 and 
                  abs(new_shoulder - shoulder_angle) < 2)
        
        return aligned
    
    def pickup_sequence(self):
        """Complete pickup sequence with YOUR tested angles"""
        if self.emergency_stopped:
            return False
            
        print("\n🍅 Starting pickup sequence...")
        
        # Phase 1: Move to pickup position
        self.move_to_position('pickup', smooth=True)
        time.sleep(1)
        
        # Phase 2: Open gripper if not already
        self.servos['gripper'].set_angle(100, smooth=True)
        time.sleep(0.5)
        
        # Phase 3: Approach tomato (small forward movement)
        current_shoulder = self.servos['shoulder'].get_angle()
        self.servos['shoulder'].set_angle(current_shoulder + 5, smooth=True)
        time.sleep(0.5)
        
        # Phase 4: Close gripper
        self.servos['gripper'].set_angle(15, smooth=True)  # YOUR tested close angle
        time.sleep(1)
        
        # Phase 5: Lift slightly
        self.move_to_position('lift', smooth=True)
        time.sleep(0.5)
        
        # Phase 6: Retract
        self.move_to_position('retract', smooth=True)
        time.sleep(0.5)
        
        # Phase 7: Move to release position
        self.move_to_position('release', smooth=True)
        time.sleep(1)
        
        # Phase 8: Open gripper to drop
        self.servos['gripper'].set_angle(100, smooth=True)
        time.sleep(1)
        
        # Phase 9: Return home
        self.move_to_position('home', smooth=True)
        
        print("✅ Pickup sequence complete!")
        return True
    
    def emergency_stop(self):
        """Emergency stop"""
        self.emergency_stopped = True
        print("🚨 EMERGENCY STOP ACTIVATED!")
        # Stop all servos
        for servo in self.servos.values():
            servo.set_angle(servo.get_angle(), smooth=False)
    
    def reset_emergency(self):
        """Reset emergency stop"""
        self.emergency_stopped = False
        print("✅ Emergency reset")
    
    def cleanup(self):
        """Clean up resources"""
        self.move_to_position('home')
        time.sleep(1)
        self.pca.deinit()

# ==========================================
# ULTRASONIC SENSOR
# ==========================================
class UltrasonicSensor:
    def __init__(self, trig_pin=23, echo_pin=24):
        self.current_distance = 0
        self.distance_history = deque(maxlen=5)
        self.running = True
        
        try:
            self.sensor = DistanceSensor(echo=echo_pin, trigger=trig_pin,
                                        max_distance=4, threshold_distance=0.1)
            print("✅ Ultrasonic sensor initialized")
        except Exception as e:
            print(f"⚠️ Ultrasonic sensor error: {e}")
            self.sensor = None
        
        self.thread = threading.Thread(target=self._continuous_reading)
        self.thread.daemon = True
        self.thread.start()
    
    def _continuous_reading(self):
        while self.running:
            if self.sensor:
                try:
                    distance = self.sensor.distance * 100  # Convert to cm
                    if 2 < distance < 400:
                        self.distance_history.append(distance)
                        if len(self.distance_history) > 2:
                            sorted_dists = sorted(self.distance_history)
                            self.current_distance = sorted_dists[len(sorted_dists)//2]
                except:
                    pass
            time.sleep(0.05)
    
    def get_distance(self):
        return self.current_distance
    
    def cleanup(self):
        self.running = False
        if self.sensor:
            self.sensor.close()

# ==========================================
# YOLO TOMATO DETECTION
# ==========================================
class TomatoDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_h = None
        self.input_w = None
        self.confidence_threshold = 0.25
        self.ripe_class_id = 2  # Adjust based on your model
        
        self.load_model()
    
    def load_model(self):
        try:
            print(f"Loading model from {self.model_path}...")
            self.interpreter = Interpreter(model_path=self.model_path, num_threads=4)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_h = self.input_details[0]['shape'][1]
            self.input_w = self.input_details[0]['shape'][2]
            
            print(f"✅ Model loaded - Input size: {self.input_w}x{self.input_h}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect(self, frame):
        """Detect ripe tomatoes in frame"""
        if self.interpreter is None:
            return []
        
        # Prepare image
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        output = output.T
        
        # Process detections
        detections = []
        orig_h, orig_w = frame.shape[:2]
        
        for pred in output:
            x, y, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = class_scores[class_id]
            
            if confidence > self.confidence_threshold and class_id == self.ripe_class_id:
                xmin = int((x - w/2) * orig_w)
                ymin = int((y - h/2) * orig_h)
                xmax = int((x + w/2) * orig_w)
                ymax = int((y + h/2) * orig_h)
                cx = int(x * orig_w)
                cy = int(y * orig_h)
                
                detections.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'center': (cx, cy),
                    'confidence': float(confidence),
                    'class_id': class_id
                })
        
        # Apply NMS
        if detections:
            boxes = [[d['bbox'][0], d['bbox'][1], 
                     d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]] 
                     for d in detections]
            scores = [d['confidence'] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, 0.45)
            
            if len(indices) > 0:
                return [detections[i] for i in indices.flatten()]
        
        return []

# ==========================================
# CAMERA STREAM
# ==========================================
class CameraStream:
    def __init__(self):
        self.cap = None
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = True
        
    def start(self):
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.output_frame = frame.copy()
                return frame
        return None
    
    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('semi_auto_control.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, 
                                          [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'state': robot.current_state,
        'emergency_stopped': robot.emergency_stopped,
        'auto_align': robot.auto_align,
        'distance': ultrasonic.get_distance(),
        'positions': {name: servo.get_angle() for name, servo in robot.servos.items()}
    })

@app.route('/api/move/<position>', methods=['POST'])
def api_move(position):
    if robot.emergency_stopped:
        return jsonify({'error': 'Emergency stop active'}), 403
    
    if position in robot.positions:
        threading.Thread(target=robot.move_to_position, args=(position, True)).start()
        return jsonify({'status': 'moving', 'position': position})
    return jsonify({'error': 'Invalid position'}), 400

@app.route('/api/pick', methods=['POST'])
def api_pick():
    if robot.emergency_stopped:
        return jsonify({'error': 'Emergency stop active'}), 403
    
    if robot.current_state == 'ready_to_pick':
        threading.Thread(target=robot.pickup_sequence).start()
        robot.current_state = 'picking'
        return jsonify({'status': 'picking'})
    return jsonify({'error': 'Not ready to pick'}), 400

@app.route('/api/emergency', methods=['POST'])
def api_emergency():
    robot.emergency_stop()
    return jsonify({'status': 'emergency_stop'})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    robot.reset_emergency()
    robot.move_to_position('home')
    return jsonify({'status': 'reset'})

@app.route('/api/toggle_align', methods=['POST'])
def api_toggle_align():
    robot.auto_align = not robot.auto_align
    return jsonify({'auto_align': robot.auto_align})

# ==========================================
# SOCKETIO EVENTS
# ==========================================
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {
        'state': robot.current_state,
        'auto_align': robot.auto_align
    })

@socketio.on('align_to_tomato')
def handle_align(data):
    if robot.auto_align and not robot.emergency_stopped:
        x_offset = data.get('x', 0)
        y_offset = data.get('y', 0)
        aligned = robot.align_to_tomato(x_offset, y_offset)
        
        if aligned:
            robot.current_state = 'ready_to_pick'
            emit('aligned', {'ready': True})
        else:
            emit('aligning', {'x': x_offset, 'y': y_offset})

# ==========================================
# MAIN PROCESSING THREAD
# ==========================================
def process_frame():
    """Main processing loop for detection and alignment"""
    global current_detections
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        
        # Create working copy
        display_frame = frame.copy()
        
        # Detect tomatoes
        detections = detector.detect(frame)
        current_detections = detections
        
        # Draw detections
        target_zone = None
        closest_tomato = None
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            conf = det['confidence']
            
            # Check if in center zone (for auto-alignment)
            frame_h, frame_w = frame.shape[:2]
            center_zone_x = frame_w // 2
            center_zone_y = frame_h // 2
            zone_size = 100
            
            in_center = (abs(cx - center_zone_x) < zone_size//2 and
                        abs(cy - center_zone_y) < zone_size//2)
            
            # Draw bounding box
            color = (0, 255, 0) if in_center else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display_frame, (cx, cy), 5, color, -1)
            
            # Add confidence text
            cv2.putText(display_frame, f"Tomato {conf:.2f}", 
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Find closest tomato to center
            if in_center and not closest_tomato:
                closest_tomato = det
                target_zone = (center_zone_x, center_zone_y, zone_size)
        
        # Draw target zone
        if target_zone:
            cx, cy, size = target_zone
            cv2.rectangle(display_frame, 
                         (cx - size//2, cy - size//2),
                         (cx + size//2, cy + size//2),
                         (255, 255, 255), 2)
            cv2.putText(display_frame, "TARGET ZONE", 
                       (cx - 40, cy - size//2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Auto-align to closest tomato if enabled
        if closest_tomato and robot.auto_align and not robot.emergency_stopped:
            cx, cy = closest_tomato['center']
            frame_h, frame_w = frame.shape[:2]
            center_x = frame_w // 2
            center_y = frame_h // 2
            
            x_offset = cx - center_x
            y_offset = cy - center_y
            
            # Only align if offset is significant
            if abs(x_offset) > 20 or abs(y_offset) > 20:
                robot.align_to_tomato(x_offset, y_offset)
            else:
                if robot.current_state != 'ready_to_pick':
                    robot.current_state = 'ready_to_pick'
                    cv2.putText(display_frame, "READY TO PICK - Press PICK button", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
        
        # Add status overlay
        status_color = (0, 0, 255) if robot.emergency_stopped else \
                      (0, 255, 0) if robot.current_state == 'ready_to_pick' else \
                      (255, 255, 0)
        
        status_text = "EMERGENCY STOP" if robot.emergency_stopped else \
                     "READY TO PICK" if robot.current_state == 'ready_to_pick' else \
                     "AUTO-ALIGN ACTIVE" if robot.auto_align else "MANUAL MODE"
        
        cv2.putText(display_frame, f"Status: {status_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display_frame, f"Auto-Align: {'ON' if robot.auto_align else 'OFF'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Distance: {ultrasonic.get_distance():.1f}cm", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Tomatoes: {len(detections)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Update output frame
        with camera.lock:
            camera.output_frame = display_frame
        
        time.sleep(0.03)

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    # Initialize components
    robot = RobotArm()
    camera = CameraStream()
    ultrasonic = UltrasonicSensor()
    
    # Load YOLO model (update path to your model)
    MODEL_PATH = "home/rslvpi5/tomato-detection/tomato-classification/best_float16.tflite"
    detector = TomatoDetector(MODEL_PATH)
    
    # Start camera
    if not camera.start():
        print("⚠️ Camera not available - running in simulation mode")
    
    # Start camera update thread
    def camera_update():
        while True:
            camera.update_frame()
            time.sleep(0.03)
    
    cam_thread = threading.Thread(target=camera_update, daemon=True)
    cam_thread.start()
    
    # Start processing thread
    process_thread = threading.Thread(target=process_frame, daemon=True)
    process_thread.start()
    
    # Move to home
    print("\n🚀 Semi-Automatic Tomato Harvester Starting...")
    robot.move_to_position('home')
    
    print("\n📡 Web interface: http://raspberrypi.local:5001")
    print("\n🎮 Controls:")
    print("   - Robot automatically detects and aligns to tomatoes")
    print("   - Click 'Pick Tomato' when ready")
    print("   - Emergency stop available")
    print("\nPress Ctrl+C to exit\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        camera.stop()
        ultrasonic.cleanup()
        robot.cleanup()
        print("✅ Shutdown complete")
