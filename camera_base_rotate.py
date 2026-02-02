import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# ==========================================
# PCA9685 SETUP
# ==========================================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# ==========================================
# CHANNEL MAPPING
# ==========================================
BASE_CH   = 0    # Arm base servo
CAMERA_CH = 6    # Camera SG90 servo

base_servo = servo.Servo(
    pca.channels[BASE_CH],
    min_pulse=500,
    max_pulse=2500
)

camera_servo = servo.Servo(
    pca.channels[CAMERA_CH],
    min_pulse=500,
    max_pulse=2500
)

# ==========================================
# SCAN LIMITS (same as your working test)
# ==========================================
SCAN_MIN = 20
SCAN_MAX = 50
SCAN_DELAY = 0.05   # matches your ServoKit test speed

# ==========================================
# MAIN PROGRAM
# ==========================================
try:
    print("🔄 Rotating BASE and CAMERA together")

    while True:
        # Move both from 20° → 50°
        for angle in range(SCAN_MIN, SCAN_MAX + 1):
            base_servo.angle = angle
            camera_servo.angle = angle
            print(f"Base: {angle}°, Camera: {angle}°")
            time.sleep(SCAN_DELAY)

        time.sleep(1)

        # Move both from 50° → 20°
        for angle in range(SCAN_MAX, SCAN_MIN - 1, -1):
            base_servo.angle = angle
            camera_servo.angle = angle
            print(f"Base: {angle}°, Camera: {angle}°")
            time.sleep(SCAN_DELAY)

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Program stopped")

finally:
    # Release servos
    base_servo.angle = None
    camera_servo.angle = None
    pca.deinit()
