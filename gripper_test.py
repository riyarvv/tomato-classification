import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# --- CONFIGURATION ---
GRIPPER_CHANNEL = 5  # <--- CHANGE THIS to your actual gripper channel
MIN_PULSE = 600      # Slightly narrowed for MG90S safety
MAX_PULSE = 2400     # Slightly narrowed for MG90S safety
MAX_ANGLE = 180

# --- INITIALIZATION ---
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Initialize the gripper servo
gripper = servo.Servo(pca.channels[GRIPPER_CHANNEL], 
                      actuation_range=MAX_ANGLE, 
                      min_pulse=MIN_PULSE, 
                      max_pulse=MAX_PULSE)

def test_gripper():
    print(f"Testing Gripper on Channel {GRIPPER_CHANNEL}...")
    try:
        while True:
            print("Closing (0 degrees)...")
            gripper.angle = 30
            time.sleep(2)
            
            print("Middle (90 degrees)...")
            gripper.angle = 90
            time.sleep(2)
            
            print("Opening (180 degrees)...")
            gripper.angle = 160
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nTest stopped.")
    finally:
        pca.deinit()

if __name__ == "__main__":
    test_gripper()
