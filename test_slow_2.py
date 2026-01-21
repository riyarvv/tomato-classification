import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# =============================
# INITIALIZATION & MAPPING
# =============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Mapping Joints to PCA Channels (Adjust indices if your wiring is different)
BASE_CH     = 0
SHOULDER_CH = 1
ELBOW_CH    = 2
PITCH_CH    = 3
GRIPPER_CH  = 4

# Initialize Servo Objects
servos = {}
channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH]
for ch in channels:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# =============================
# CONFIGURATION LIMITS
# =============================
# Format: (Neutral, Min_Limit, Max_Limit)
LIMITS = {
    BASE_CH:     {"neutral": 30,  "min": 20,  "max": 40},
    PITCH_CH:    {"neutral": 90,  "min": 40,  "max": 120}, # Updated range
    SHOULDER_CH: {"neutral": 130, "pick": 115},
    ELBOW_CH:    {"neutral": 65,  "pick": 100},
    GRIPPER_CH:  {"open": 180,    "close": 0}
}

# =============================
# MOVEMENT LOGIC
# =============================
def move_slow(channel_id, target_angle, speed=0.04):
    """Moves a servo degree-by-degree to prevent jerking."""
    current = servos[channel_id].angle
    
    # If starting for the first time, assume neutral to avoid sudden snap
    if current is None:
        current = LIMITS[channel_id].get("neutral", 90)
    
    start_angle = int(current)
    target_angle = int(target_angle)
    
    # Safety Clamp
    if "min" in LIMITS[channel_id] and "max" in LIMITS[channel_id]:
        target_angle = max(LIMITS[channel_id]["min"], min(LIMITS[channel_id]["max"], target_angle))

    if start_angle == target_angle:
        return

    step = 1 if target_angle > start_angle else -1
    for angle in range(start_angle, target_angle + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

def go_home():
    print("Moving to Neutral/Home Position...")
    # Move in a specific order to avoid hitting the base
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_tomato():
    print("Executing Pick Sequence...")
    # 1. Position Base and Pitch
    move_slow(BASE_CH, 40)
    move_slow(PITCH_CH, 90)
    
    # 2. Extend Arm
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    
    # 3. Close Gripper (Faster speed for grip)
    print("Gripping...")
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], speed=0.01)
    time.sleep(1)
    
    # 4. Retract
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    go_home()

# =============================
# MAIN EXECUTION
# =============================
if __name__ == "__main__":
    try:
        go_home()
        time.sleep(2)
        pick_tomato()
    except KeyboardInterrupt:
        print("\nEmergency Stop Triggered.")
    finally:
        pca.deinit()
