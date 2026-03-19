from flask import Flask, Response, redirect, jsonify, render_template_string
import serial
import threading
import subprocess
import cv2
import time
import os
import signal

app = Flask(__name__)

# ================================
# SERIAL CONNECTION (ESP32)
# ================================
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    print("✅ Serial connection established")
except:
    print("⚠️ Could not connect to serial. Make sure ESP32 is connected.")
    ser = None

# ================================
# GLOBAL VARIABLES
# ================================
harvesting = False
tomato_count = 0
scan_process = None
robot_status = {
    'state': 'idle',
    'current_target': None,
    'distance': 0,
    'fps': 0
}

# ================================
# SERIAL LISTENER THREAD
# ================================
def read_serial():
    global tomato_count, robot_status

    while True:
        if ser and ser.in_waiting:
            try:
                line = ser.readline().decode().strip()
                print(f"📟 Serial: {line}")

                if line.startswith("COUNT:"):
                    tomato_count = int(line.split(":")[1])
                    print(f"📊 Tomato count updated: {tomato_count}")
                
                elif line.startswith("STATUS:"):
                    # Parse status from ESP32 if needed
                    pass
                    
            except Exception as e:
                print(f"Serial error: {e}")
        time.sleep(0.01)

# ================================
# MAIN CONTROL PAGE
# ================================
@app.route("/control")
def control():
    return """
<html>
<head>
<title>Agribot Controller - Smart Harvester</title>
<style>
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0;
        padding: 20px;
        color: white;
    }
    
    .container {
        max-width: 800px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        color: #333;
    }
    
    h1 {
        color: #4CAF50;
        margin-top: 0;
    }
    
    h2 {
        color: #555;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    .video {
        border: 5px solid #4CAF50;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    
    .control-panel {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        max-width: 400px;
        margin: 20px auto;
    }
    
    button {
        width: 100%;
        height: 70px;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        background: #4CAF50;
        color: white;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    button:hover {
        background: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    button:active {
        transform: translateY(0);
    }
    
    button.stop {
        background: #f44336;
    }
    
    button.stop:hover {
        background: #da190b;
    }
    
    button.harvest {
        background: #ff9800;
    }
    
    button.harvest:hover {
        background: #f57c00;
    }
    
    .speed-slider {
        width: 100%;
        margin: 20px 0;
    }
    
    .countbox {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
        background: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .status-panel {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 20px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .status-item {
        text-align: center;
    }
    
    .status-label {
        font-size: 14px;
        color: #666;
    }
    
    .status-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .status-value.warning {
        color: #f44336;
    }
    
    .btn-group {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .btn-group button {
        width: auto;
        min-width: 120px;
        height: 50px;
    }
</style>
</head>
<body>
    <div class="container">
        <h1>🤖 Smart Tomato Harvester</h1>
        
        <h2>Live Camera Feed</h2>
        <img src="/video_feed" class="video" id="videoFeed" 
             onerror="this.src='https://via.placeholder.com/640x480?text=Connecting...'">
        
        <div class="status-panel">
            <div class="status-item">
                <div class="status-label">Robot State</div>
                <div class="status-value" id="state">Idle</div>
            </div>
            <div class="status-item">
                <div class="status-label">Distance</div>
                <div class="status-value" id="distance">0 cm</div>
            </div>
            <div class="status-item">
                <div class="status-label">FPS</div>
                <div class="status-value" id="fps">0</div>
            </div>
        </div>
        
        <h2>Base Movement</h2>
        <div class="control-panel">
            <div></div>
            <button onclick="sendCommand('F')">⬆ FORWARD</button>
            <div></div>
            <button onclick="sendCommand('L')">⬅ LEFT</button>
            <button onclick="sendCommand('S')">⏹️ STOP</button>
            <button onclick="sendCommand('R')">➡ RIGHT</button>
            <div></div>
            <button onclick="sendCommand('B')">⬇ BACK</button>
            <div></div>
        </div>
        
        <h2>Motor Speed</h2>
        <input type="range" min="0" max="255" value="180" 
               class="speed-slider" onchange="setSpeed(this.value)"
               oninput="document.getElementById('speedVal').innerText = this.value">
        <div>Speed: <span id="speedVal">180</span></div>
        
        <h2>Harvesting Control</h2>
        <div class="btn-group">
            <button class="harvest" onclick="startHarvest()">▶️ START HARVEST</button>
            <button class="stop" onclick="stopHarvest()">⏹️ STOP HARVEST</button>
        </div>
        
        <div class="countbox">
            🍅 Tomatoes Harvested: <span id="count">0</span>
        </div>
        
        <div class="btn-group">
            <button onclick="resetCount()">🔄 Reset Counter</button>
            <button onclick="returnHome()">🏠 Return to Home</button>
            <button class="stop" onclick="emergencyStop()">⚠️ EMERGENCY STOP</button>
        </div>
        
        <p style="margin-top: 20px; color: #888; font-size: 12px;">
            Smart Harvester v2.0 - With Inverse Kinematics
        </p>
    </div>
    
    <script>
        function sendCommand(cmd) {
            fetch('/move/' + cmd)
                .then(response => response.text())
                .then(data => console.log('Command sent:', cmd, data))
                .catch(err => console.error('Error:', err));
        }
        
        function setSpeed(val) {
            fetch('/speed/' + val)
                .then(() => console.log('Speed set to:', val));
        }
        
        function startHarvest() {
            fetch('/pick')
                .then(() => {
                    document.getElementById('state').innerText = 'HARVESTING';
                    document.getElementById('state').style.color = '#ff9800';
                })
                .catch(err => console.error('Error:', err));
        }
        
        function stopHarvest() {
            fetch('/stop_harvest')
                .then(() => {
                    document.getElementById('state').innerText = 'IDLE';
                    document.getElementById('state').style.color = '#4CAF50';
                })
                .catch(err => console.error('Error:', err));
        }
        
        function resetCount() {
            fetch('/reset')
                .then(() => updateCount())
                .catch(err => console.error('Error:', err));
        }
        
        function returnHome() {
            fetch('/home')
                .then(() => console.log('Returning home...'))
                .catch(err => console.error('Error:', err));
        }
        
        function emergencyStop() {
            fetch('/emergency')
                .then(() => {
                    document.getElementById('state').innerText = 'EMERGENCY STOP';
                    document.getElementById('state').style.color = '#f44336';
                })
                .catch(err => console.error('Error:', err));
        }
        
        function updateCount() {
            fetch('/count')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('count').innerText = data.count;
                })
                .catch(err => console.error('Error:', err));
        }
        
        function updateStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('state').innerText = data.state.toUpperCase();
                    document.getElementById('distance').innerText = data.distance + ' cm';
                    document.getElementById('fps').innerText = data.fps;
                    
                    if(data.state === 'picking') {
                        document.getElementById('state').style.color = '#ff9800';
                    } else if(data.state === 'scanning') {
                        document.getElementById('state').style.color = '#4CAF50';
                    } else {
                        document.getElementById('state').style.color = '#f44336';
                    }
                })
                .catch(err => console.log('Status update error'));
        }
        
        // Update every second
        setInterval(() => {
            updateCount();
            updateStatus();
        }, 1000);
    </script>
</body>
</html>
"""

# ================================
# VIDEO STREAM
# ================================
@app.route('/video_feed')
def video_feed():
    return redirect("http://raspberrypi.local:5001/video_feed")

# ================================
# API STATUS ENDPOINT
# ================================
@app.route("/api/status")
def api_status():
    """Get current robot status from scan_pick.py if running"""
    global robot_status, scan_process
    
    # If scan_pick is running, try to get its status
    if scan_process and scan_process.poll() is None:
        # You could implement a pipe or file-based communication here
        # For now, we'll just return basic info
        robot_status['state'] = 'picking' if harvesting else 'scanning'
    
    return jsonify(robot_status)

# ================================
# COUNT ROUTES
# ================================
@app.route("/count")
def get_count():
    return jsonify({"count": tomato_count})

@app.route("/increment")
def increment_count():
    global tomato_count

    tomato_count += 1

    # send updated count to ESP32
    if ser:
        ser.write(f"COUNT:{tomato_count}\n".encode())

    print(f"🍅 Tomato Harvested! Total: {tomato_count}")

    return jsonify({"count": tomato_count})

@app.route("/reset")
def reset():
    global tomato_count

    tomato_count = 0
    if ser:
        ser.write(b'Z')

    print("🔄 Counter reset to 0")

    return jsonify({"status": "reset"})

# ================================
# START HARVESTING
# ================================
@app.route('/pick')
def pick():
    global scan_process, harvesting, robot_status

    print("🚀 Start Harvest button pressed")

    # Check if already running
    if scan_process and scan_process.poll() is None:
        print("⚠️ Harvesting already running")
        return jsonify({"status": "already_running"})

    try:
        # Kill any existing process
        if scan_process:
            scan_process.terminate()
            time.sleep(1)
        
        # Start new scan_pick.py process
        scan_process = subprocess.Popen(
            [
                "/home/rslvpi5/tomato-detection/tomato-classification/venv/bin/python",
                "kinematics.py"
            ],
            cwd="/home/rslvpi5/tomato-detection/tomato-classification",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        harvesting = True
        robot_status['state'] = 'scanning'
        print("✅ scan_pick.py started successfully")

        return jsonify({"status": "started"})

    except Exception as e:
        print(f"❌ Error starting harvest: {e}")
        return jsonify({"status": "error", "message": str(e)})

# ================================
# STOP HARVESTING
# ================================
@app.route('/stop_harvest')
def stop_harvest():
    global scan_process, harvesting, robot_status

    print("🛑 Stop Harvest button pressed")

    harvesting = False
    robot_status['state'] = 'idle'

    if scan_process:
        try:
            # Try graceful termination first
            scan_process.terminate()
            time.sleep(2)
            
            # Force kill if still running
            if scan_process.poll() is None:
                scan_process.kill()
            
            scan_process = None
            print("✅ scan_pick.py stopped")

            # Send stop command to base movement
            if ser:
                ser.write(b'S')

        except Exception as e:
            print(f"❌ Error stopping harvest: {e}")

    return jsonify({"status": "stopped"})

# ================================
# RETURN TO HOME POSITION
# ================================
@app.route('/home')
def home_position():
    """Return arm to home position"""
    global scan_process
    
    print("🏠 Returning to home position...")
    
    if scan_process and scan_process.poll() is None:
        # If scan_pick is running, we need to send home command
        # This could be implemented via a file or signal
        # For now, we'll just stop and restart
        stop_harvest()
        time.sleep(1)
    
    # Send home command via serial to ESP32 if needed
    if ser:
        ser.write(b'H')
    
    return jsonify({"status": "returning_home"})

# ================================
# EMERGENCY STOP
# ================================
@app.route('/emergency')
def emergency():
    """Emergency stop - stops all movement"""
    global scan_process, harvesting
    
    print("🚨 EMERGENCY STOP ACTIVATED!")
    
    harvesting = False
    
    # Stop scan_pick.py
    if scan_process:
        try:
            scan_process.terminate()
            time.sleep(1)
            if scan_process.poll() is None:
                scan_process.kill()
            scan_process = None
        except:
            pass
    
    # Send emergency stop to ESP32
    if ser:
        ser.write(b'E')  # Emergency stop command
    
    # Also send stop command to be sure
    if ser:
        ser.write(b'S')
    
    return jsonify({"status": "emergency_stop_activated"})

# ================================
# ROBOT MOVEMENT
# ================================
@app.route('/move/<cmd>')
def move(cmd):
    global scan_process, harvesting

    cmd = cmd.upper()
    print(f"🎮 Movement command: {cmd}")

    # If harvesting is active, stop it when manual movement is requested
    if cmd != "S" and harvesting:
        print("🔄 Manual movement requested - stopping harvest")
        stop_harvest()

    if ser:
        if cmd == "F":
            ser.write(b'F')
            return "Forward"
        elif cmd == "B":
            ser.write(b'B')
            return "Back"
        elif cmd == "L":
            ser.write(b'L')
            return "Left"
        elif cmd == "R":
            ser.write(b'R')
            return "Right"
        elif cmd == "S":
            harvesting = False
            ser.write(b'S')
            
            if scan_process:
                scan_process.terminate()
                scan_process = None
            
            return "Stop"
    else:
        return "Serial not connected"

    return "Invalid Command"

# ================================
# MOTOR SPEED CONTROL
# ================================
@app.route('/speed/<int:value>')
def set_speed(value):
    value = max(0, min(255, value))

    if ser:
        command = f"SPEED:{value}\n"
        ser.write(command.encode())
        print(f"⚡ Speed set to: {value}")

    return jsonify({"speed": value})

# ================================
# HOME PAGE
# ================================
@app.route("/")
def home():
    return redirect("/control")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    # Start serial listener thread
    if ser:
        serial_thread = threading.Thread(target=read_serial, daemon=True)
        serial_thread.start()

    print("\n" + "="*50)
    print("🚀 AGRIBOT CONTROLLER SERVER")
    print("="*50)
    print("📡 Web interface: http://raspberrypi.local:5000")
    print("📡 Video stream: http://raspberrypi.local:5001/video_feed")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
