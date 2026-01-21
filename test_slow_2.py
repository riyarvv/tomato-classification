import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# =============================
# INITIALIZATION
# =============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# DOUBLE CHECK THESE CHANNEL NUMBERS:
# If gripper_test.py worked, check which channel index you used there!
BASE_CH     = 0
SHOULDER_CH = 1
ELBOW_CH    = 2
PITCH_CH    = 3
GRIPPER_CH  = 5

servos = {}
channels = [BASE_CH, SHOULDER_CH, ELBOW_CH, PITCH_CH, GRIPPER_CH]
for ch in channels:
    servos[ch] = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)

# =============================
# CONFIGURATION
# =============================
LIMITS = {
    BASE_CH:     {"neutral": 20,  "min": 10,  "max": 50},
    PITCH_CH:    {"neutral": 90,  "min": 40,  "max": 120},
    SHOULDER_CH: {"neutral": 130, "pick": 115},
    ELBOW_CH:    {"neutral": 65,  "pick": 100},
    GRIPPER_CH:  {"open": 160,    "close": 30} # Changed 0 to 10 to prevent stalling
}

def move_slow(channel_id, target_angle, speed=0.04):
    current = servos[channel_id].angle
    if current is None:
        current = 90 # Default starting assumption
    
    start_angle = int(current)
    target_angle = int(target_angle)
    
    # Simple safety clamp
    target_angle = max(0, min(180, target_angle))

    if start_angle == target_angle:
        return

    step = 1 if target_angle > start_angle else -1
    for angle in range(start_angle, target_angle + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

# =============================
# MOVEMENT SEQUENCES
# =============================

def go_home():
    print("Resetting to Neutral Position...")
    # Open gripper first to release anything held
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(PITCH_CH, LIMITS[PITCH_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])

def pick_tomato():
    print("Moving to Pick...")
    move_slow(BASE_CH, 40)
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])
    
    print("Gripping...")
    # We move to 'close' limit. 
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["close"], speed=0.02)
    time.sleep(1)

def drop_tomato():
    print("Executing Drop Sequence...")
    # 1. Return to Neutral first (so it drops in the collection bin)
    move_slow(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_slow(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])
    move_slow(BASE_CH, LIMITS[BASE_CH]["neutral"])
    
    # 2. Release Gripper
    print("Releasing Gripper...")
    move_slow(GRIPPER_CH, LIMITS[GRIPPER_CH]["open"], speed=0.02)
    print("Drop Complete.")

# =============================
# MAIN RUN
# =============================
if __name__ == "__main__":
    try:
        go_home()
        time.sleep(1)
        pick_tomato()
        time.sleep(1)
        drop_tomato()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pca.deinit()
