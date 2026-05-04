from flask import Flask, Response, jsonify,redirect
import serial
import threading
import subprocess
import cv2

app = Flask(__name__)

# ================================
# SERIAL CONNECTIONS
# ================================
esp = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# ================================
# CAMERA INIT (STREAM INSIDE SERVER)
# ================================

# ================================
# GLOBAL VARIABLES
# ================================
harvesting = False
tomato_count = 0
scan_process = None

# ================================
# SERIAL LISTENER (ESP COUNT)
# ================================
def read_serial():
    global tomato_count

    while True:
        if esp.in_waiting:
            line = esp.readline().decode().strip()

            if line.startswith("COUNT:"):
                tomato_count = int(line.split(":")[1])
                print("🍅 Updated Count:", tomato_count)

# ================================
# VIDEO STREAM GENERATOR
# ================================


# ================================
# CONTROL PAGE
# ================================
@app.route("/control")
def control():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
  <title>Agribot Pro | Smart Farming Controller</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(145deg, #0a1f1a 0%, #0c2b23 100%);
      min-height: 100vh;
      padding: 24px 28px;
      color: #e9f5ef;
    }

    /* main layout */
    .dashboard {
      max-width: 1600px;
      margin: 0 auto;
    }

    h1 {
      font-size: 1.9rem;
      font-weight: 600;
      letter-spacing: -0.3px;
      background: linear-gradient(135deg, #d4ffb0, #7ee0b0);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      display: inline-flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 28px;
    }

    h1::before {
      content: "🌾";
      font-size: 2rem;
      background: none;
      -webkit-background-clip: unset;
      color: #c0ffb0;
    }

    .grid-container {
      display: grid;
      grid-template-columns: 2.2fr 1.2fr;
      gap: 24px;
    }

    /* cards modern style */
    .card {
      background: rgba(18, 32, 28, 0.75);
      backdrop-filter: blur(2px);
      border-radius: 32px;
      border: 1px solid rgba(90, 150, 120, 0.35);
      box-shadow: 0 20px 35px -12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
      padding: 22px 24px;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .card:hover {
      box-shadow: 0 24px 40px -14px rgba(0, 0, 0, 0.5);
      border-color: rgba(110, 200, 150, 0.5);
    }

    .section-title {
      font-weight: 600;
      font-size: 1.25rem;
      letter-spacing: -0.2px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-left: 4px solid #6bcb8c;
      padding-left: 14px;
      color: #e0ffe6;
    }

    .video-container {
      position: relative;
      background: #0a1612;
      border-radius: 24px;
      overflow: hidden;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }

    .video {
      width: 100%;
      display: block;
      border-radius: 24px;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #0f201b;
      border: 1px solid #2c5542;
    }

    /* movement grid */
    .movement-panel {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-bottom: 28px;
    }

    .dpad {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      margin: 15px 0;
    }

    .dpad-row {
      display: flex;
      justify-content: center;
      gap: 18px;
    }

    .ctrl-btn {
      background: rgba(30, 50, 44, 0.9);
      border: none;
      width: 80px;
      height: 80px;
      border-radius: 48px;
      font-size: 2.2rem;
      font-weight: 600;
      cursor: pointer;
      color: #d4ffd8;
      backdrop-filter: blur(4px);
      box-shadow: 0 6px 0 #0e2a20;
      transition: all 0.08s linear;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: monospace;
    }

    .ctrl-btn:active {
      transform: translateY(4px);
      box-shadow: 0 2px 0 #0e2a20;
    }

    .ctrl-btn.small {
      width: 70px;
      height: 70px;
      font-size: 1.8rem;
    }

    .stop-btn {
      background: #8b3c2c;
      box-shadow: 0 6px 0 #542012;
      color: #ffe0cf;
    }

    /* speed slider */
    .speed-section {
      background: #11231e;
      border-radius: 24px;
      padding: 16px 20px;
      margin: 20px 0;
    }

    .slider-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      font-weight: 500;
      margin-bottom: 12px;
      letter-spacing: 0.3px;
      color: #b4e6cf;
    }

    input[type="range"] {
      width: 100%;
      height: 6px;
      -webkit-appearance: none;
      background: #2b5242;
      border-radius: 10px;
      outline: none;
    }

    input[type="range"]:focus {
      outline: none;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 20px;
      height: 20px;
      background: #7be0a8;
      border-radius: 50%;
      cursor: pointer;
      box-shadow: 0 0 6px #96f0bc;
      border: none;
    }

    .speed-value {
      font-weight: 700;
      background: #1f3d34;
      padding: 4px 12px;
      border-radius: 40px;
      font-size: 0.85rem;
    }

    /* harvest buttons group */
    .harvest-group {
      display: flex;
      gap: 20px;
      justify-content: center;
      margin: 20px 0;
    }

    .action-btn {
      background: linear-gradient(105deg, #236b4c, #1a543b);
      border: none;
      padding: 12px 28px;
      border-radius: 60px;
      font-weight: 600;
      font-size: 1rem;
      color: #f0fff0;
      cursor: pointer;
      transition: all 0.2s;
      font-family: 'Inter', sans-serif;
      letter-spacing: 0.3px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .action-btn.harvest-start {
      background: linear-gradient(105deg, #2c8e5e, #1f6e48);
    }

    .action-btn.harvest-stop {
      background: linear-gradient(105deg, #b9583c, #9a4128);
    }

    .action-btn.reset-btn {
      background: #2f4b44;
      box-shadow: none;
    }

    .action-btn:hover {
      filter: brightness(1.05);
      transform: translateY(-2px);
    }

    /* counter badge */
    .count-card {
      background: #0a1f18;
      border-radius: 28px;
      text-align: center;
      padding: 18px 12px;
      margin: 20px 0 12px;
      border: 1px solid #358062;
    }

    .count-label {
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 2px;
      font-weight: 400;
      color: #9ccfb6;
    }

    .count-number {
      font-size: 3.7rem;
      font-weight: 800;
      line-height: 1;
      margin: 10px 0;
      color: #f3ffcf;
      text-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .unit {
      font-size: 1rem;
      font-weight: 400;
      color: #a7dbbc;
    }

    .reset-btn-full {
      width: 100%;
      background: #253e36;
      margin-top: 6px;
      justify-content: center;
    }

    hr {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, #2c5e4a, #4f8b70, #2c5e4a);
      margin: 16px 0;
    }

    .status-badge {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #0e241f;
      border-radius: 48px;
      padding: 8px 18px;
      margin-top: 16px;
      font-size: 0.8rem;
    }

    .led {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: #6bdc9c;
      box-shadow: 0 0 6px #6bdc9c;
      display: inline-block;
      margin-right: 8px;
    }

    @media (max-width: 880px) {
      body {
        padding: 16px;
      }
      .grid-container {
        grid-template-columns: 1fr;
        gap: 20px;
      }
      .ctrl-btn {
        width: 65px;
        height: 65px;
        font-size: 1.8rem;
      }
    }

    button {
      cursor: pointer;
      user-select: none;
    }
  </style>
</head>
<body>
<div class="dashboard">
  <h1>Agribot Controller · Precision Agriculture</h1>

  <div class="grid-container">
    <!-- LEFT: VIDEO FEED -->
    <div class="card">
      <div class="section-title">
        <span>📡</span> Live Field View
      </div>
      <div class="video-container">
        <img src="/video_feed" class="video" alt="agribot camera feed" id="videoFeed" onerror="this.src='https://placehold.co/800x450/1e3a32/6fcf97?text=Connecting+to+camera...'">
      </div>
      <div class="status-badge">
        <span><span class="led"></span> RTSP Stream Active</span>
        <span>🔄 real‑time</span>
      </div>
    </div>

    <!-- RIGHT: CONTROLS & HARVEST -->
    <div class="card">
      <!-- Movement Section -->
      <div class="section-title">
        <span>🎮</span> Navigation System
      </div>
      <div class="movement-panel">
        <div class="dpad">
          <div class="dpad-row">
            <button class="ctrl-btn" data-cmd="F" aria-label="Forward">▲</button>
          </div>
          <div class="dpad-row">
            <button class="ctrl-btn" data-cmd="L" aria-label="Left">◀</button>
            <button class="ctrl-btn stop-btn" data-cmd="S" aria-label="Stop">■</button>
            <button class="ctrl-btn" data-cmd="R" aria-label="Right">▶</button>
          </div>
          <div class="dpad-row">
            <button class="ctrl-btn" data-cmd="B" aria-label="Backward">▼</button>
          </div>
        </div>
      </div>

      <!-- Speed Control -->
      <div class="speed-section">
        <div class="slider-label">
          <span>⚡ Motor Speed (PWM)</span>
          <span class="speed-value" id="speedDisplay">180</span>
        </div>
        <input type="range" id="speedSlider" class="slider" min="0" max="255" value="180" step="1">
        <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.7rem; color: #8abfaa;">
          <span>Slow</span><span>Agile</span><span>Max</span>
        </div>
      </div>

      <hr>

      <!-- HARVEST MODULE -->
      <div class="section-title">
        <span>🍅</span> Harvest Manager
      </div>
      <div class="harvest-group">
        <button class="action-btn harvest-start" id="startHarvestBtn">🌱 Start Harvesting</button>
        <button class="action-btn harvest-stop" id="stopHarvestBtn">⏹️ Stop</button>
      </div>

      <!-- Tomato Counter -->
      <div class="count-card">
        <div class="count-label">Total Yield</div>
        <div class="count-number">
          <span id="count">0</span><span class="unit">  fruits</span>
        </div>
        <div style="font-size: 0.75rem; color: #bee9d4;">🍅 premium tomatoes collected</div>
      </div>

      <button class="action-btn reset-btn reset-btn-full" id="resetCountBtn">
        🔄 Reset Counter
      </button>

      <hr>

      <div style="display: flex; justify-content: center; gap: 10px; margin-top: 8px;">
        <div style="font-size: 0.7rem; background: #142f27; padding: 5px 12px; border-radius: 40px;">
          🤖 Agribot v2.0 · ROS ready
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  // ---------- API endpoints (assumed backend) ----------
  // Movement: GET /move/<cmd>   (F, B, L, R, S)
  // Speed:    GET /speed/<value>
  // Harvest:  GET /pick          (start harvest)
  // Stop:     GET /stop_harvest
  // Count:    GET /count         returns { count: int }
  // Reset:    GET /reset

  // Helper: send command with fetch, error handling (silent but console)
  function sendCommand(endpoint) {
    fetch(endpoint).catch(err => console.warn(`Command failed: ${endpoint}`, err));
  }

  // Movement buttons handler
  const moveButtons = document.querySelectorAll('.ctrl-btn');
  moveButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const cmd = btn.getAttribute('data-cmd');
      if (cmd) {
        // visual feedback: add mini active effect
        btn.style.transform = 'translateY(2px)';
        btn.style.boxShadow = '0 2px 0 #0e2a20';
        setTimeout(() => {
          btn.style.transform = '';
          btn.style.boxShadow = '';
        }, 120);
        sendCommand(`/move/${cmd}`);
      }
    });
  });

  // Speed slider
  const speedSlider = document.getElementById('speedSlider');
  const speedDisplay = document.getElementById('speedDisplay');

  function updateSpeed(value) {
    const val = parseInt(value, 10);
    speedDisplay.innerText = val;
    sendCommand(`/speed/${val}`);
  }

  speedSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    speedDisplay.innerText = val;
  });

  speedSlider.addEventListener('change', (e) => {
    updateSpeed(e.target.value);
  });

  // Harvest buttons
  const startHarvestBtn = document.getElementById('startHarvestBtn');
  const stopHarvestBtn = document.getElementById('stopHarvestBtn');

  startHarvestBtn.addEventListener('click', () => {
    sendCommand('/pick');
    // subtle haptic-like feedback
    startHarvestBtn.style.transform = 'scale(0.97)';
    setTimeout(() => startHarvestBtn.style.transform = '', 120);
  });

  stopHarvestBtn.addEventListener('click', () => {
    sendCommand('/stop_harvest');
    stopHarvestBtn.style.transform = 'scale(0.97)';
    setTimeout(() => stopHarvestBtn.style.transform = '', 120);
  });

  // Reset count
  const resetBtn = document.getElementById('resetCountBtn');
  resetBtn.addEventListener('click', () => {
    sendCommand('/reset');
    resetBtn.style.transform = 'scale(0.97)';
    setTimeout(() => resetBtn.style.transform = '', 150);
    // optimistic UI update (fetch after 100ms)
    setTimeout(() => updateCount(), 80);
  });

  // ----- Counter update with smooth retry & JSON handling -----
  async function updateCount() {
    try {
      const response = await fetch('/count');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      // safety: ensure data.count is number
      const newCount = (data && typeof data.count === 'number') ? data.count : (data.count ? parseInt(data.count,10) : 0);
      document.getElementById('count').innerText = newCount;
    } catch (error) {
      console.debug("Count fetch error, keeping previous value", error);
      // optional: show placeholder, but we don't reset UI
    }
  }

  // Refresh counter every 800ms for snappy feedback
  let counterInterval = setInterval(updateCount, 800);

  // Also add realtime video reload helper (optional: force refresh if needed, but browser handles /video_feed)
  // In case image fails due to connection, attempt to reload image source after error
  const videoImg = document.getElementById('videoFeed');
  if (videoImg) {
    videoImg.addEventListener('error', () => {
      // retry to reload stream every 3 seconds if broken (optional)
      if (!videoImg.hasAttribute('data-retry')) {
        videoImg.setAttribute('data-retry', '1');
        const retryInterval = setInterval(() => {
          if (videoImg.complete && (videoImg.naturalWidth === 0 || videoImg.naturalHeight === 0)) {
            videoImg.src = '/video_feed?_t=' + Date.now();
          } else {
            clearInterval(retryInterval);
          }
        }, 2500);
      }
    });
  }

  // extra: show speed initial value on load
  window.addEventListener('load', () => {
    updateSpeed(speedSlider.value);
    updateCount();
    // Send idle keep-alive not required, but ensures initial sync
  });

  // optional: keyboard controls for professional use (WASD + arrows + space stop)
  function handleKeyControls(e) {
    const key = e.key;
    let cmd = null;
    // Arrow keys + WASD standard
    if (key === 'ArrowUp' || key === 'w' || key === 'W') cmd = 'F';
    else if (key === 'ArrowLeft' || key === 'a' || key === 'A') cmd = 'L';
    else if (key === 'ArrowRight' || key === 'd' || key === 'D') cmd = 'R';
    else if (key === 'ArrowDown' || key === 's' || key === 'S') cmd = 'B';
    else if (key === ' ' || key === 'Space' || key === 'Stop' || key === 'x' || key === 'X') {
      cmd = 'S';
      e.preventDefault();  // prevent page scrolling on space
    } else if (key === '0' || key === 'Escape') {
      cmd = 'S';
    }

    if (cmd) {
      e.preventDefault();
      sendCommand(`/move/${cmd}`);
      // give visual feedback on corresponding button if exists
      const btn = document.querySelector(`.ctrl-btn[data-cmd="${cmd}"]`);
      if (btn) {
        btn.style.transform = 'translateY(2px)';
        btn.style.boxShadow = '0 2px 0 #0e2a20';
        setTimeout(() => {
          if(btn) {
            btn.style.transform = '';
            btn.style.boxShadow = '';
          }
        }, 130);
      }
    }
  }

  window.addEventListener('keydown', handleKeyControls);

  // optional speed shortcuts: +/- to adjust speed quickly
  function handleSpeedShortcuts(e) {
    if (e.key === '+' || e.key === '=' || e.key === 'ArrowUp' && e.ctrlKey) {
      e.preventDefault();
      let newVal = parseInt(speedSlider.value, 10) + 10;
      newVal = Math.min(255, newVal);
      speedSlider.value = newVal;
      updateSpeed(newVal);
      speedDisplay.innerText = newVal;
    } else if (e.key === '-' || e.key === '_' || e.key === 'ArrowDown' && e.ctrlKey) {
      e.preventDefault();
      let newVal = parseInt(speedSlider.value, 10) - 10;
      newVal = Math.max(0, newVal);
      speedSlider.value = newVal;
      updateSpeed(newVal);
      speedDisplay.innerText = newVal;
    }
  }
  window.addEventListener('keydown', handleSpeedShortcuts);

  // Cleanup interval on page unload (optional but good)
  window.addEventListener('beforeunload', () => {
    if (counterInterval) clearInterval(counterInterval);
  });

  // For additional robustness: add touchstart events to buttons to avoid delay on mobile
  const allBtns = document.querySelectorAll('button');
  allBtns.forEach(btn => {
    btn.addEventListener('touchstart', (e) => {
      // just to trigger active style, but no double fire
      if (btn.classList.contains('ctrl-btn')) {
        const cmd = btn.getAttribute('data-cmd');
        if (cmd) sendCommand(`/move/${cmd}`);
      } else if (btn.id === 'startHarvestBtn') {
        sendCommand('/pick');
      } else if (btn.id === 'stopHarvestBtn') {
        sendCommand('/stop_harvest');
      } else if (btn.id === 'resetCountBtn') {
        sendCommand('/reset');
        setTimeout(() => updateCount(), 80);
      }
      e.preventDefault();
    });
  });
</script>
</body>
</html>
"""

# ================================
# VIDEO FEED (NOW LOCAL)
# ================================
@app.route('/video_feed')
def video_feed():
    return redirect("http://10.215.117.125:5002/video_feed")

# ================================
# COUNT ROUTES
# ================================
@app.route("/count")
def get_count():
    return jsonify({"count": tomato_count})

@app.route("/increment")
def increment():
    global tomato_count

    tomato_count += 1
    esp.write(f"COUNT:{tomato_count}\n".encode())

    print("🍅 Tomato Count:", tomato_count)

    return {"count": tomato_count}

@app.route("/reset")
def reset():
    global tomato_count
    tomato_count = 0
    esp.write(b'Z\n')
    return {"status": "reset"}

# ================================
# START HARVESTING
# ================================
@app.route('/pick')
def pick():
    global scan_process, harvesting

    if scan_process is None:
        scan_process = subprocess.Popen(
            ["python3", "tom.py"]
        )
        harvesting = True
        print("🌱 Harvesting started")
        return "Harvesting Started"

    return "Already Running"

# ================================
# STOP HARVESTING
# ================================
@app.route('/stop_harvest')
def stop_harvest():
    global scan_process, harvesting

    harvesting = False

    if scan_process:
        scan_process.terminate()
        scan_process = None

    return "Stopped"

# ================================
# ROBOT MOVEMENT (ESP)
# ================================
@app.route('/move/<cmd>')
def move(cmd):
    cmd = cmd.upper()

    if cmd in ['F','B','L','R','S']:
        esp.write((cmd + '\n').encode())
        return f"Sent {cmd}"

    return "Invalid"

# ================================
# SPEED CONTROL (ESP)
# ================================
@app.route('/speed/<int:value>')
def speed(value):
    value = max(0, min(255, value))
    esp.write(f"SPEED:{value}\n".encode())
    return {"speed": value}

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    threading.Thread(target=read_serial, daemon=True).start()

    print("🚀 Server Running at http://<PI-IP>:5001/control")

    app.run(host="0.0.0.0", port=5001, threaded=True,use_reloader=False)
