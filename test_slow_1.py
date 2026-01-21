import time
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio

# ==============================
# PCA9685 SETUP
# ==============================
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==============================
# CHANNEL DEFINITIONS
# ==============================
BASE = 0
SHOULDER = 1
ELBOW = 2
PITCH = 3
GRIPPER = 4

# ==============================
# SERVO LIMITS (SAFE)
# ==============================
SERVO_MIN = 500
SERVO_MAX = 2500

def angle_to_duty(angle):
    pulse = SERVO_MIN + (angle / 180.0) * (SERVO_MAX - SERVO_MIN)
    return int((pulse / 20000) * 65535)

def set_angle(ch, angle):
    angle = max(0, min(180, angle))
    pca.channels[ch].duty_cycle = angle_to_duty(angle)

# ==============================
# SLOW SINGLE-SERVO MOVE
# ==============================
def slow_move(ch, start, end, step=1, delay=0.03):
    if start < end:
        for a in range(start, end + 1, step):
            set_angle(ch, a)
            time.sleep(delay)
    else:
        for a in range(start, end - 1, -step):
            set_angle(ch, a)
            time.sleep(delay)

# ==============================
# HOME POSITION
# ==============================
def home():
    print("Moving to HOME")

    set_angle(PITCH, 90)
    time.sleep(0.5)

    slow_move(GRIPPER, 180, 0)
    time.sleep(0.5)

    slow_move(ELBOW, 100, 65)
    time.sleep(0.5)

    slow_move(SHOULDER, 115, 130)
    time.sleep(0.5)

    slow_move(BASE, 90, 0)
    time.sleep(0.5)

# ==============================
# PICK SEQUENCE (ONE BY ONE)
# ==============================
def pick():
    print("Picking tomato")

    # 1. Open gripper
    slow_move(GRIPPER, 0, 180)
    time.sleep(0.5)

    # 2. Shoulder moves down
    slow_move(SHOULDER, 130, 115)
    time.sleep(0.5)

    # 3. Elbow moves forward
    slow_move(ELBOW, 65, 100)
    time.sleep(0.5)

    # 4. Close gripper
    slow_move(GRIPPER, 180, 0)
    time.sleep(0.5)

# ==============================
# DROP SEQUENCE (ONE BY ONE)
# ==============================
def drop():
    print("Dropping tomato")

    # 1. Lift arm
    slow_move(ELBOW, 100, 65)
    time.sleep(0.5)

    slow_move(SHOULDER, 115, 130)
    time.sleep(0.5)

    # 2. Rotate to cart
    slow_move(BASE, 0, 90)
    time.sleep(0.5)

    # 3. Release
    slow_move(GRIPPER, 0, 180)
    time.sleep(0.5)

# ==============================
# MAIN
# ==============================
try:
    home()
    time.sleep(1)

    pick()
    time.sleep(1)

    drop()
    time.sleep(1)

    home()

except KeyboardInterrupt:
    pca.deinit()
