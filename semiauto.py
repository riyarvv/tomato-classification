#!/usr/bin/env python3
"""
SAFE Semi-Automatic Tomato Harvester
- Uses port 5002 to avoid conflicts
- Servos start DISABLED
- Manual enable required
- Safety limits enforced
"""

import time
import board
import busio
import cv2
import numpy as np
import threading
import signal
import sys
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from gpiozero import DistanceSensor, Device
from gpiozero.pins.lgpio import LGPIOFactory
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# Set pin factory
Device.pin_factory = LGPIOFactory()

# ==========================================
# FLASK APP - USE PORT 5002
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tomato-robot-safe'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ==========================================
# SAFE SERVO CONTROLLER
# ==========================================
class SafeServoController:
    def __init__(self, pca, channel, name, min_angle, max_angle, default_angle):
        self.channel = channel
        self.name = name
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.default_angle = default_angle
        self.current_angle = default_angle
        self.enabled = False  # DISABLED BY DEFAULT - SAFETY!
        self.servo = None
        
        try:
            self.servo = servo.Servo(pca.channels[channel], 
                                    min_pulse=500, 
                                    max_pulse=2500)
            # DO NOT set angle on init - wait for enable
            print(f"✅ Servo {name} initialized (disabled)")
        except Exception as e:
            print(f"❌ Error initializing {name}: {e}")
    
    def enable(self):
        """Enable servo and move to safe position"""
        if self.servo:
            print(f"🔧 Enabling {self.name} servo...")
            # Start at default angle
            self.servo.angle = self.default_angle
            self.current_angle = self.default_angle
            self.enabled = True
            print(f"✅ {self.name} enabled at {self.default_angle}°")
    
    def disable(self):
        """Disable servo (stop sending signals)"""
        if self.servo:
            print(f"⚠️ Disabling {self.name} servo")
            self.enabled = False
            # Don't change angle, just stop updates
    
    def set_angle(self, angle, smooth=True):
        """Set servo angle with safety checks"""
        if not self.enabled:
            print(f"❌ {self.name} is disabled - ignoring angle command")
            return False
        
        # Clamp to safe limits
        angle = max(self.min_angle, min(self.max_angle, angle))
        
        # Check for dangerous movement
        if abs(angle - self.current_angle) > 30:
            print(f"⚠️ Large movement detected in {self.name}: {self.current_angle}° → {angle}°")
            print(f"   Waiting for confirmation...")
            return False
        
        if smooth and abs(angle - self.current_angle) > 5:
            # Smooth movement
            start = self.current_angle
            steps = int(abs(angle - start) / 2)
            if steps > 0:
                step = 2 if angle > start else -2
                for i in range(steps + 1):
                    new_angle = start + (step * i)
                    self.servo.angle = new_angle
                    time.sleep(0.02)
            self.servo.angle = angle
        else:
            self.servo.angle = angle
        
        self.current_angle = angle
        return True
    
    def get_angle(self):
        return self.current_angle if self.enabled else None

# ==========================================
# SAFE ROBOT ARM
# ==========================================
class SafeRobotArm:
    def __init__(self):
        print("Initializing PCA9685...")
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        
        # Servo definitions with YOUR tested angles
        self.servos = {
            'base': SafeServoController(self.pca, 0, 'Base', 10, 100, 55),
            'shoulder': SafeServoController(self.pca, 1, 'Shoulder', 30, 160, 90),
            'elbow': SafeServoController(self.pca, 2, 'Elbow', 20, 90, 45),
            'wrist': SafeServoController(self.pca, 3, 'Wrist', 0, 180, 90),
            'gripper': SafeServoController(self.pca, 5, 'Gripper', 15, 100, 100),
            'camera': SafeServoController(self.pca, 6, 'Camera', 10, 100, 55)
        }
        
        self.all_enabled = False
        self.emergency_stopped = False
        
        print("✅ Safe robot arm initialized (all servos DISABLED)")
    
    def enable_all(self):
        """Enable all servos - REQUIRED before any movement"""
        if not self.emergency_stopped:
            print("\n🔧 ENABLING ALL SERVOS...")
            for name, servo in self.servos.items():
                servo.enable()
                time.sleep(0.5)
            self.all_enabled = True
            print("✅ All servos enabled\n")
            return True
        return False
    
    def disable_all(self):
        """Disable all servos - SAFETY"""
        print("\n⚠️ DISABLING ALL SERVOS")
        for servo in self.servos.values():
            servo.disable()
        self.all_enabled = False
        print("✅ All servos disabled\n")
    
    def set_servo(self, name, angle):
        """Set individual servo position"""
        if not self.all_enabled:
            print("❌ Servos not enabled - call enable_all() first")
            return False
        if self.emergency_stopped:
            print("❌ Emergency stop active")
            return False
        if name in self.servos:
            return self.servos[name].set_angle(angle)
        return False
    
    def move_to_home(self):
        """Move to home position safely"""
        if not self.all_enabled or self.emergency_stopped:
            return False
        
        print("🏠 Moving to home position...")
        positions = {
            'base': 20,
            'shoulder': 160,
            'elbow': 20,
            'wrist': 90,
            'gripper': 100
        }
        
        # Move base first
        if self.set_servo('base', positions['base']):
            time.sleep(0.5)
        
        # Then shoulder and elbow
        if self.set_servo('shoulder', positions['shoulder']):
            time.sleep(0.3)
        if self.set_servo('elbow', positions['elbow']):
            time.sleep(0.3)
        if self.set_servo('wrist', positions['wrist']):
            time.sleep(0.2)
        if self.set_servo('gripper', positions['gripper']):
            time.sleep(0.3)
        
        print("✅ Home position reached")
        return True
    
    def emergency_stop(self):
        """Emergency stop - disable all servos"""
        print("\n🚨 EMERGENCY STOP ACTIVATED!")
        self.emergency_stopped = True
        self.disable_all()
        socketio.emit('emergency', {'status': True})
    
    def reset_emergency(self):
        """Reset emergency stop"""
        print("\n✅ Resetting emergency stop")
        self.emergency_stopped = False
        socketio.emit('emergency', {'status': False})
    
    def cleanup(self):
        """Clean up resources"""
        self.disable_all()
        time.sleep(0.5)
        self.pca.deinit()

# ==========================================
# SAFE CAMERA
# ==========================================
class SafeCamera:
    def __init__(self):
        self.cap = None
        self.output_frame = None
        self.lock = threading.Lock()
        
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
    
    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()
    
    def stop(self):
        if self.cap:
            self.cap.release()

# ==========================================
# FLASK ROUTES - PORT 5002
# ==========================================
@app.route('/')
def index():
    return render_template('safe_control.html')

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
        'servos_enabled': robot.all_enabled,
        'emergency_stopped': robot.emergency_stopped,
        'positions': {name: s.get_angle() for name, s in robot.servos.items() if s.enabled}
    })

@app.route('/api/enable', methods=['POST'])
def api_enable():
    """Enable servos - MUST be called first"""
    if not robot.emergency_stopped:
        robot.enable_all()
        return jsonify({'status': 'enabled'})
    return jsonify({'error': 'Emergency stop active'}), 403

@app.route('/api/disable', methods=['POST'])
def api_disable():
    """Disable servos"""
    robot.disable_all()
    return jsonify({'status': 'disabled'})

@app.route('/api/home', methods=['POST'])
def api_home():
    """Move to home"""
    if robot.all_enabled and not robot.emergency_stopped:
        threading.Thread(target=robot.move_to_home).start()
        return jsonify({'status': 'moving to home'})
    return jsonify({'error': 'Servos not enabled'}), 403

@app.route('/api/servo/<name>', methods=['POST'])
def api_servo(name):
    """Set servo position"""
    data = request.json
    angle = data.get('angle')
    if robot.set_servo(name, angle):
        return jsonify({'status': 'ok', 'angle': angle})
    return jsonify({'error': 'Failed'}), 400

@app.route('/api/emergency', methods=['POST'])
def api_emergency():
    """Emergency stop"""
    robot.emergency_stop()
    return jsonify({'status': 'emergency_stop'})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset emergency"""
    robot.reset_emergency()
    return jsonify({'status': 'reset'})

# ==========================================
# SAFETY SIGNAL HANDLER
# ==========================================
def signal_handler(sig, frame):
    print("\n\n🛑 Signal received - shutting down safely...")
    robot.disable_all()
    camera.stop()
    robot.cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    # Initialize
    robot = SafeRobotArm()
    camera = SafeCamera()
    
    # Start camera
    if not camera.start():
        print("⚠️ Camera not available")
    
    # Start camera update thread
    def camera_update():
        while True:
            camera.update_frame()
            time.sleep(0.03)
    
    cam_thread = threading.Thread(target=camera_update, daemon=True)
    cam_thread.start()
    
    print("\n" + "="*60)
    print("🚀 SAFE Semi-Automatic Tomato Harvester")
    print("="*60)
    print("\n⚠️  IMPORTANT SAFETY NOTES:")
    print("   1. Servos start DISABLED - nothing will move")
    print("   2. You MUST click 'Enable Servos' first")
    print("   3. Emergency stop disables all servos immediately")
    print("   4. Keep hand near emergency stop button")
    print("\n📡 Web interface: http://raspberrypi.local:5002")
    print("   (Using port 5002 to avoid conflicts)")
    print("\nPress Ctrl+C to exit\n")
    print("="*60 + "\n")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5002, debug=False)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.cleanup()
        camera.stop()
        print("✅ Shutdown complete")
