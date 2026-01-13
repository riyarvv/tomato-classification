import time
import board
import busio
from adafruit_pca9685 import PCA9685

# =============================
# PCA9685 Initialization
# =============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50   # Standard servo frequency

# =============================
# Servo angle function
# =============================
def set_servo_angle(channel, angle):
    angle = max(30, min(150, angle))   # safe limits
    pulse = int(4096 * (0.5 + angle / 180 * 2.0) / 20)
    pca.channels[channel].duty_cycle = pulse

# =============================
# Servo channels
# =============================
BASE = 0
SHOULDER = 1
ELBOW = 2
WRIST = 3
WRIST_ROTATE = 4
GRIPPER = 5

# =============================
# Neutral position
# =============================
print("Moving all servos to neutral position")
set_servo_angle(BASE, 90)
set_servo_angle(SHOULDER, 90)
set_servo_angle(ELBOW, 90)
set_servo_angle(WRIST, 90)
set_servo_angle(WRIST_ROTATE, 90)
set_servo_angle(GRIPPER, 45)
time.sleep(3)

# =============================
# Test each servo separately
# =============================
def test_servo(name, channel):
    print(f"Testing {name}")
    for angle in [80, 90, 100]:
        set_servo_angle(channel, angle)
        time.sleep(1)
    set_servo_angle(channel, 90)
    time.sleep(1)

test_servo("BASE", BASE)
test_servo("SHOULDER", SHOULDER)
test_servo("ELBOW", ELBOW)
test_servo("WRIST", WRIST)
test_servo("WRIST ROTATE", WRIST_ROTATE)
test_servo("GRIPPER", GRIPPER)

print("Servo testing completed safely")

pca.deinit()
