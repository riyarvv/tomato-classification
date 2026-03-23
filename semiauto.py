#!/usr/bin/env python3
"""
FIXED Robot Control - Single File Version
Fixed variable scope issues
"""

import time
import threading
from flask import Flask, Response, jsonify, request, render_template_string

# Flask app
app = Flask(__name__)

# Try to import hardware modules
HARDWARE_AVAILABLE = False
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    HARDWARE_AVAILABLE = True
    print("✅ Hardware modules loaded")
except ImportError as e:
    print(f"⚠️ Hardware modules not available: {e}")
    print("   Running in SIMULATION mode")

# Try to import camera
CAMERA_AVAILABLE = False
try:
    import cv2
    import numpy as np
    CAMERA_AVAILABLE = True
    print("✅ Camera modules loaded")
except ImportError as e:
    print(f"⚠️ Camera modules not available: {e}")

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
    
    @property
    def angle(self):
        return self.current_angle
    
    @angle.setter
    def angle(self, value):
        self.set_angle(value)

# Robot controller
class SimpleRobot:
    def __init__(self):
        self.servos = {}
        self.all_enabled = False
        self.emergency_stopped = False
        self.pca = None
        self.hardware_available = HARDWARE_AVAILABLE  # Use global variable
        
        if self.hardware_available:
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
                self.hardware_available = False
                self._init_simulation()
        else:
            self._init_simulation()
    
    def _init_simulation(self):
        """Initialize simulated servos"""
        print("⚠️ Using SIMULATED servos (no hardware)")
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
            # Clamp angle
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
    
    def get_status(self):
        return {
            'servos_enabled': self.all_enabled,
            'emergency_stopped': self.emergency_stopped,
            'positions': {name: self.servos[name]['current'] for name in self.servos}
        }
    
    def cleanup(self):
        if self.pca:
            self.pca.deinit()
            print("✅ PCA9685 cleaned up")

# Camera class
class SimpleCamera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = True
        self.available = CAMERA_AVAILABLE
    
    def start(self):
        if not self.available:
            print("⚠️ Camera not available - simulation mode")
            return False
        
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print("✅ Camera initialized")
                return True
            else:
                print("❌ Cannot open camera")
                self.available = False
                return False
        except Exception as e:
            print(f"❌ Camera error: {e}")
            self.available = False
            return False
    
    def update(self):
        if self.available and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
    
    def get_frame(self):
        if not self.available:
            # Create a blank frame with text
            if CAMERA_AVAILABLE:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera Not Available", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return blank
            return None
        return self.frame
    
    def stop(self):
        if self.cap:
            self.cap.release()
            print("✅ Camera released")

# HTML template (embedded)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Arm Control</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
            margin: 0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 2em;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }
        .card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            color: #333;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .video-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            text-align: center;
        }
        img {
            width: 100%;
            height: auto;
            display: block;
        }
        button {
            padding: 12px 20px;
            margin: 5px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-enable {
            background: #4caf50;
            color: white;
            font-size: 18px;
            padding: 15px;
            width: 100%;
        }
        .btn-disable {
            background: #f44336;
            color: white;
            width: 100%;
        }
        .btn-home {
            background: #2196f3;
            color: white;
            width: 100%;
        }
        .btn-emergency {
            background: #ff0000;
            color: white;
            font-size: 18px;
            padding: 15px;
            width: 100%;
            margin-top: 10px;
        }
        .slider {
            width: 100%;
            margin: 10px 0;
        }
        .status {
            background: #f5f5f5;
            padding: 12px;
            margin: 10px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .led {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .led-red {
            background: #f44336;
            box-shadow: 0 0 5px #f44336;
        }
        .led-green {
            background: #4caf50;
            box-shadow: 0 0 5px #4caf50;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }
        .warning {
            background: #ff9800;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
            color: white;
        }
        .info {
            background: #2196f3;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 12px;
            text-align: center;
        }
        hr {
            margin: 20px 0;
            border: none;
            border-top: 2px solid #ddd;
        }
        .servo-control {
            margin: 15px 0;
        }
        .servo-control label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #1e3c72;
        }
        .angle-value {
            display: inline-block;
            background: #1e3c72;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Tomato Harvesting Robot Control</h1>
        
        <div class="warning" id="warningMsg">
            ⚠️ SERVOS START DISABLED - Click "ENABLE SERVOS" first
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📷 Camera Feed</h2>
                <div class="video-container">
                    <img id="video" src="/video_feed" alt="Camera Feed">
                </div>
                <div class="info" id="cameraInfo">
                    Camera: <span id="cameraStatus">Loading...</span>
                </div>
            </div>
            
            <div>
                <div class="card">
                    <h2>🎮 Main Controls</h2>
                    
                    <div class="status">
                        <span><span class="led" id="statusLed"></span> Servo Status:</span>
                        <span id="servoStatus">DISABLED</span>
                    </div>
                    <div class="status">
                        <span>Emergency Stop:</span>
                        <span id="emergencyStatus">INACTIVE</span>
                    </div>
                    <div class="status">
                        <span>Mode:</span>
                        <span id="modeStatus">SIMULATION</span>
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
                    
                    <h3>⚙️ Individual Servo Control</h3>
                    
                    <div class="servo-control">
                        <label>Base Rotation (10-100°)</label>
                        <input type="range" id="base" class="slider" min="10" max="100" step="1" value="55" disabled>
                        <span id="baseVal" class="angle-value">55°</span>
                    </div>
                    
                    <div class="servo-control">
                        <label>Shoulder (30-160°)</label>
                        <input type="range" id="shoulder" class="slider" min="30" max="160" step="1" value="90" disabled>
                        <span id="shoulderVal" class="angle-value">90°</span>
                    </div>
                    
                    <div class="servo-control">
                        <label>Elbow (20-90°)</label>
                        <input type="range" id="elbow" class="slider" min="20" max="90" step="1" value="45" disabled>
                        <span id="elbowVal" class="angle-value">45°</span>
                    </div>
                    
                    <div class="servo-control">
                        <label>Gripper (15-100°)</label>
                        <input type="range" id="gripper" class="slider" min="15" max="100" step="1" value="100" disabled>
                        <span id="gripperVal" class="angle-value">100°</span>
                    </div>
                    
                    <button id="emergencyBtn" class="btn-emergency" onclick="emergencyStop()">
                        🚨 EMERGENCY STOP
                    </button>
                    <button onclick="resetEmergency()" style="background:#666; color:white; width:100%; margin-top:5px;">
                        🔄 Reset Emergency
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        var servosEnabled = false;
        var hardwareMode = false;
        
        // Setup sliders
        function setupSlider(id) {
            $('#' + id).on('input', function() {
                var angle = parseInt($(this).val());
                $('#' + id + 'Val').text(angle + '°');
                if (servosEnabled) {
                    $.post('/api/servo/' + id, JSON.stringify({angle: angle}), 
                        function(data) {
                            console.log(id + ' moved to ' + angle);
                        })
                    .fail(function() {
                        console.log('Failed to move ' + id);
                    });
                }
            });
        }
        
        setupSlider('base');
        setupSlider('shoulder');
        setupSlider('elbow');
        setupSlider('gripper');
        
        function enableServos() {
            $.post('/api/enable', function(data) {
                servosEnabled = true;
                $('#servoStatus').text('ENABLED').css('color', '#4caf50');
                $('#statusLed').removeClass('led-red').addClass('led-green');
                $('#enableBtn').prop('disabled', true);
                $('#disableBtn').prop('disabled', false);
                $('#homeBtn').prop('disabled', false);
                $('.slider').prop('disabled', false);
                $('#warningMsg').html('✅ SERVOS ENABLED - Moving to home position');
                setTimeout(function() {
                    $('#warningMsg').html('✅ Servos active - Use controls carefully');
                }, 3000);
            }).fail(function() {
                alert('Failed to enable servos');
            });
        }
        
        function disableServos() {
            $.post('/api/disable', function() {
                servosEnabled = false;
                $('#servoStatus').text('DISABLED').css('color', '#f44336');
                $('#statusLed').removeClass('led-green').addClass('led-red');
                $('#enableBtn').prop('disabled', false);
                $('#disableBtn').prop('disabled', true);
                $('#homeBtn').prop('disabled', true);
                $('.slider').prop('disabled', true);
                $('#warningMsg').html('⚠️ Servos disabled - Click "ENABLE SERVOS" to activate');
            });
        }
        
        function moveHome() {
            if (servosEnabled) {
                $.post('/api/home', function() {
                    console.log('Moving home');
                });
            }
        }
        
        function emergencyStop() {
            if (confirm('⚠️ EMERGENCY STOP! This will disable all servos. Continue?')) {
                $.post('/api/emergency', function() {
                    servosEnabled = false;
                    $('#servoStatus').text('EMERGENCY STOP').css('color', '#ff0000');
                    $('#emergencyStatus').text('ACTIVE').css('color', '#ff0000');
                    $('#statusLed').removeClass('led-green').addClass('led-red');
                    $('#enableBtn').prop('disabled', true);
                    $('#disableBtn').prop('disabled', true);
                    $('#homeBtn').prop('disabled', true);
                    $('.slider').prop('disabled', true);
                    $('#warningMsg').html('🚨 EMERGENCY STOP ACTIVE - Click "Reset Emergency" to recover');
                });
            }
        }
        
        function resetEmergency() {
            $.post('/api/reset', function() {
                $('#emergencyStatus').text('INACTIVE').css('color', '#4caf50');
                $('#enableBtn').prop('disabled', false);
                $('#warningMsg').html('⚠️ Emergency reset - Click "ENABLE SERVOS" to continue');
            });
        }
        
        function refreshStatus() {
            $.get('/api/status', function(data) {
                if (data.servos_enabled) {
                    servosEnabled = true;
                    $('#servoStatus').text('ENABLED').css('color', '#4caf50');
                    $('#statusLed').removeClass('led-red').addClass('led-green');
                } else {
                    servosEnabled = false;
                    $('#servoStatus').text('DISABLED').css('color', '#f44336');
                    $('#statusLed').removeClass('led-green').addClass('led-red');
                }
                
                if (data.emergency_stopped) {
                    $('#emergencyStatus').text('ACTIVE').css('color', '#ff0000');
                } else {
                    $('#emergencyStatus').text('INACTIVE').css('color', '#4caf50');
                }
                
                if (data.mode === 'hardware') {
                    hardwareMode = true;
                    $('#modeStatus').text('HARDWARE').css('color', '#4caf50');
                } else {
                    $('#modeStatus').text('SIMULATION').css('color', '#ff9800');
                }
                
                $('#cameraStatus').text(data.camera ? 'Active' : 'Not Available');
            });
        }
        
        setInterval(refreshStatus, 2000);
        refreshStatus();
    </script>
</body>
</html>
"""

# Initialize robot and camera
print("\n" + "="*60)
print("Initializing Robot Control System...")
print("="*60)

robot = SimpleRobot()
camera = SimpleCamera()

# Start camera
if CAMERA_AVAILABLE:
    camera.start()
else:
    print("⚠️ Camera module not available - video feed will show placeholder")

# Camera update thread
def camera_update_thread():
    while True:
        camera.update()
        time.sleep(0.03)

threading.Thread(target=camera_update_thread, daemon=True).start()

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is not None and CAMERA_AVAILABLE:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           buffer.tobytes() + b'\r\n')
            elif CAMERA_AVAILABLE:
                # Create a blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera Not Available", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.1)
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    status = robot.get_status()
    status['mode'] = 'hardware' if robot.hardware_available else 'simulation'
    status['camera'] = CAMERA_AVAILABLE and camera.available
    return jsonify(status)

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
        return jsonify({'status': 'ok', 'angle': angle})
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
    print("🚀 Robot Control Interface - FIXED VERSION")
    print("="*60)
    
    if robot.hardware_available:
        print("✅ HARDWARE MODE - Real servos connected")
    else:
        print("⚠️ SIMULATION MODE - No real hardware")
    
    if CAMERA_AVAILABLE:
        print("✅ Camera module loaded")
    else:
        print("⚠️ Camera module not loaded")
    
    print("\n📡 Web interface:")
    print("   http://localhost:5003")
    print("   http://raspberrypi.local:5003")
    print("\n⚠️  SAFETY INSTRUCTIONS:")
    print("   1. Servos start DISABLED - nothing will move")
    print("   2. Click 'ENABLE SERVOS' to activate")
    print("   3. Robot will move to home position when enabled")
    print("   4. Keep mouse near EMERGENCY STOP button")
    print("\nPress Ctrl+C to exit\n")
    print("="*60 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        robot.cleanup()
        camera.stop()
        print("✅ Cleanup complete")
