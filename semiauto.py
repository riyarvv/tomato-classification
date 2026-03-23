#!/usr/bin/env python3
"""
SIMPLE Robot Control - Single File Version
No external HTML files needed - everything in one file
Uses port 5003 to avoid conflicts
"""

import time
import board
import busio
import cv2
import numpy as np
import threading
from flask import Flask, Response, jsonify, request, render_template_string

# HTML template as string (no external file needed)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Control</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1e3c72;
            color: white;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.95);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            color: #333;
        }
        h1 { text-align: center; color: white; }
        h2 { color: #1e3c72; margin-top: 0; }
        .video-container {
            background: black;
            border-radius: 10px;
            overflow: hidden;
            text-align: center;
        }
        img { width: 100%; height: auto; }
        button {
            padding: 12px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .btn-enable { background: #4caf50; color: white; font-size: 18px; padding: 15px; }
        .btn-disable { background: #f44336; color: white; }
        .btn-home { background: #2196f3; color: white; }
        .btn-emergency { background: #ff0000; color: white; font-size: 18px; padding: 15px; }
        .slider { width: 100%; margin: 10px 0; }
        .status {
            background: #f5f5f5;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }
        .led {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .led-red { background: #f44336; }
        .led-green { background: #4caf50; animation: pulse 1s infinite; }
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }
        .warning {
            background: #ff9800;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍅 Robot Arm Control</h1>
        
        <div class="warning">
            ⚠️ SERVOS START DISABLED - Click "ENABLE SERVOS" first
        </div>
        
        <div class="card">
            <h2>📷 Camera Feed</h2>
            <div class="video-container">
                <img id="video" src="/video_feed">
            </div>
        </div>
        
        <div class="card">
            <h2>🎮 Controls</h2>
            
            <div class="status">
                <span>Servo Status:</span>
                <span id="servoStatus">DISABLED</span>
            </div>
            <div class="status">
                <span>Emergency:</span>
                <span id="emergencyStatus">INACTIVE</span>
            </div>
            
            <button id="enableBtn" class="btn-enable" onclick="enableServos()">
                🔌 ENABLE SERVOS
            </button>
            <button id="disableBtn" class="btn-disable" onclick="disableServos()" disabled>
                ⚡ DISABLE SERVOS
            </button>
            <button id="homeBtn" class="btn-home" onclick="moveHome()" disabled>
                🏠 MOVE TO HOME
            </button>
            
            <hr>
            
            <h3>Individual Servo Control</h3>
            <label>Base (10-100°):</label>
            <input type="range" id="base" class="slider" min="10" max="100" step="1" value="55" disabled>
            <span id="baseVal">55°</span><br>
            
            <label>Shoulder (30-160°):</label>
            <input type="range" id="shoulder" class="slider" min="30" max="160" step="1" value="90" disabled>
            <span id="shoulderVal">90°</span><br>
            
            <label>Elbow (20-90°):</label>
            <input type="range" id="elbow" class="slider" min="20" max="90" step="1" value="45" disabled>
            <span id="elbowVal">45°</span><br>
            
            <label>Gripper (15-100°):</label>
            <input type="range" id="gripper" class="slider" min="15" max="100" step="1" value="100" disabled>
            <span id="gripperVal">100°</span><br>
            
            <button id="emergencyBtn" class="btn-emergency" onclick="emergencyStop()">
                🚨 EMERGENCY STOP
            </button>
            <button onclick="resetEmergency()" style="background:#666; color:white; width:100%;">
                🔄 Reset Emergency
            </button>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        var servosEnabled = false;
        
        function setupSlider(id) {
            $('#' + id).on('input', function() {
                var angle = $(this).val();
                $('#' + id + 'Val').text(angle + '°');
                if (servosEnabled) {
                    $.post('/api/servo/' + id, JSON.stringify({angle: parseInt(angle)}), 
                        function(data) {})
                    .fail(function() { alert('Failed'); });
                }
            });
        }
        
        setupSlider('base');
        setupSlider('shoulder');
        setupSlider('elbow');
        setupSlider('gripper');
        
        function enableServos() {
            $.post('/api/enable', function() {
                servosEnabled = true;
                $('#servoStatus').text('ENABLED').css('color', '#4caf50');
                $('#enableBtn').prop('disabled', true);
                $('#disableBtn').prop('disabled', false);
                $('#homeBtn').prop('disabled', false);
                $('.slider').prop('disabled', false);
                alert('✅ Servos enabled!');
            }).fail(function() { alert('Failed to enable'); });
        }
        
        function disableServos() {
            $.post('/api/disable', function() {
                servosEnabled = false;
                $('#servoStatus').text('DISABLED').css('color', '#f44336');
                $('#enableBtn').prop('disabled', false);
                $('#disableBtn').prop('disabled', true);
                $('#homeBtn').prop('disabled', true);
                $('.slider').prop('disabled', true);
            });
        }
        
        function moveHome() {
            $.post('/api/home', function() {});
        }
        
        function emergencyStop() {
            if (confirm('⚠️ EMERGENCY STOP!')) {
                $.post('/api/emergency', function() {
                    servosEnabled = false;
                    $('#servoStatus').text('EMERGENCY').css('color', '#ff0000');
                    $('#emergencyStatus').text('ACTIVE').css('color', '#ff0000');
                    $('#enableBtn').prop('disabled', true);
                    $('#disableBtn').prop('disabled', true);
                    $('#homeBtn').prop('disabled', true);
                    $('.slider').prop('disabled', true);
                });
            }
        }
        
        function resetEmergency() {
            $.post('/api/reset', function() {
                $('#emergencyStatus').text('INACTIVE').css('color', '#4caf50');
                $('#enableBtn').prop('disabled', false);
            });
        }
        
        function refreshStatus() {
            $.get('/api/status', function(data) {
                if (data.servos_enabled) {
                    servosEnabled = true;
                    $('#servoStatus').text('ENABLED').css('color', '#4caf50');
                } else {
                    servosEnabled = false;
                    $('#servoStatus').text('DISABLED').css('color', '#f44336');
                }
                if (data.emergency_stopped) {
                    $('#emergencyStatus').text('ACTIVE').css('color', '#ff0000');
                } else {
                    $('#emergencyStatus').text('INACTIVE').css('color', '#4caf50');
                }
            });
        }
        
        setInterval(refreshStatus, 2000);
        refreshStatus();
    </script>
</body>
</html>
"""

# Flask app
app = Flask(__name__)

# Try to import hardware modules (with fallback for testing)
try:
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
    print("✅ Hardware modules loaded")
except ImportError as e:
    print(f"⚠️ Hardware not available: {e}")
    HARDWARE_AVAILABLE = False

# Simple servo simulator for testing
class SimulatedServo:
    def __init__(self, name, min_angle, max_angle):
        self.name = name
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.current_angle = 90
        self.enabled = False
    
    def set_angle(self, angle):
        self.current_angle = max(self.min_angle, min(self.max_angle, angle))
        print(f"[SIM] {self.name} -> {self.current_angle}°")
        return True
    
    def get_angle(self):
        return self.current_angle

# Robot controller
class SimpleRobot:
    def __init__(self):
        self.servos = {}
        self.all_enabled = False
        self.emergency_stopped = False
        
        if HARDWARE_AVAILABLE:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(self.i2c)
                self.pca.frequency = 50
                
                # Real servos
                self.servos = {
                    'base': {'obj': servo.Servo(self.pca.channels[0]), 'min': 10, 'max': 100, 'current': 55},
                    'shoulder': {'obj': servo.Servo(self.pca.channels[1]), 'min': 30, 'max': 160, 'current': 90},
                    'elbow': {'obj': servo.Servo(self.pca.channels[2]), 'min': 20, 'max': 90, 'current': 45},
                    'gripper': {'obj': servo.Servo(self.pca.channels[5]), 'min': 15, 'max': 100, 'current': 100},
                }
                print("✅ Real servos initialized")
            except Exception as e:
                print(f"❌ Hardware error: {e}")
                HARDWARE_AVAILABLE = False
        
        if not HARDWARE_AVAILABLE:
            # Simulated servos
            self.servos = {
                'base': {'obj': SimulatedServo('Base', 10, 100), 'min': 10, 'max': 100, 'current': 55},
                'shoulder': {'obj': SimulatedServo('Shoulder', 30, 160), 'min': 30, 'max': 160, 'current': 90},
                'elbow': {'obj': SimulatedServo('Elbow', 20, 90), 'min': 20, 'max': 90, 'current': 45},
                'gripper': {'obj': SimulatedServo('Gripper', 15, 100), 'min': 15, 'max': 100, 'current': 100},
            }
    
    def enable_all(self):
        if not self.emergency_stopped:
            self.all_enabled = True
            print("✅ All servos enabled")
            # Move to home
            self.move_to_home()
            return True
        return False
    
    def disable_all(self):
        self.all_enabled = False
        print("⚠️ All servos disabled")
    
    def set_servo(self, name, angle):
        if not self.all_enabled or self.emergency_stopped:
            return False
        if name in self.servos:
            angle = max(self.servos[name]['min'], min(self.servos[name]['max'], angle))
            self.servos[name]['obj'].angle = angle
            self.servos[name]['current'] = angle
            print(f"✅ {name} -> {angle}°")
            return True
        return False
    
    def move_to_home(self):
        if not self.all_enabled or self.emergency_stopped:
            return False
        print("🏠 Moving to home...")
        self.set_servo('base', 20)
        time.sleep(0.3)
        self.set_servo('shoulder', 160)
        time.sleep(0.3)
        self.set_servo('elbow', 20)
        time.sleep(0.3)
        self.set_servo('gripper', 100)
        print("✅ At home")
        return True
    
    def emergency_stop(self):
        self.emergency_stopped = True
        self.all_enabled = False
        print("🚨 EMERGENCY STOP")
    
    def reset_emergency(self):
        self.emergency_stopped = False
        print("✅ Emergency reset")
    
    def cleanup(self):
        if HARDWARE_AVAILABLE and hasattr(self, 'pca'):
            self.pca.deinit()

# Camera
class SimpleCamera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = True
    
    def start(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except:
            return False
    
    def update(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
    
    def get_frame(self):
        return self.frame
    
    def stop(self):
        if self.cap:
            self.cap.release()

# Initialize
robot = SimpleRobot()
camera = SimpleCamera()

# Start camera
camera.start()

# Camera update thread
def camera_thread():
    while True:
        camera.update()
        time.sleep(0.03)

threading.Thread(target=camera_thread, daemon=True).start()

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'servos_enabled': robot.all_enabled,
        'emergency_stopped': robot.emergency_stopped
    })

@app.route('/api/enable', methods=['POST'])
def api_enable():
    robot.enable_all()
    return jsonify({'status': 'enabled'})

@app.route('/api/disable', methods=['POST'])
def api_disable():
    robot.disable_all()
    return jsonify({'status': 'disabled'})

@app.route('/api/home', methods=['POST'])
def api_home():
    robot.move_to_home()
    return jsonify({'status': 'home'})

@app.route('/api/servo/<name>', methods=['POST'])
def api_servo(name):
    data = request.json
    angle = data.get('angle')
    if robot.set_servo(name, angle):
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'failed'}), 400

@app.route('/api/emergency', methods=['POST'])
def api_emergency():
    robot.emergency_stop()
    return jsonify({'status': 'emergency'})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    robot.reset_emergency()
    return jsonify({'status': 'reset'})

# Main
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Simple Robot Control - Single File Version")
    print("="*60)
    
    if HARDWARE_AVAILABLE:
        print("✅ Hardware mode - Real servos")
    else:
        print("⚠️ Simulation mode - No real hardware")
    
    print("\n📡 Web interface: http://localhost:5003")
    print("   (or http://raspberrypi.local:5003)")
    print("\n⚠️  IMPORTANT:")
    print("   1. Servos start DISABLED")
    print("   2. Click 'ENABLE SERVOS' to start")
    print("   3. Emergency stop available")
    print("\nPress Ctrl+C to exit\n")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        robot.cleanup()
        camera.stop()
        print("✅ Done")
