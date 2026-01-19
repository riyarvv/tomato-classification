import time
import board
import busio
from adafruit_pca9685 import PCA9685

# =============================
# PCA9685 Initialization
# =============================
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50   

# =============================
# Servo Configuration
# =============================
BASE, SHOULDER, ELBOW, WRIST, WRIST_ROTATE, GRIPPER = 0, 1, 2, 3, 4, 5
current_angles = {i: 90 for i in range(6)}

def set_servo_angle(channel, angle):
    angle = max(0, min(180, angle))
    # Convert 0-180 degrees to 16-bit duty cycle (approx 5% to 10% of 65535)
    # Most servos: 0.5ms (3276) to 2.5ms (16384)
    min_duty = 1638  # 0.5ms at 50Hz
    max_duty = 8192  # 2.5ms at 50Hz
    duty = int(min_duty + (angle / 180) * (max_duty - min_duty))
    pca.channels[channel].duty_cycle = duty

def move_servo_slow(channel, target_angle, speed=0.05):
    start_angle = int(current_angles[channel])
    step = 1 if target_angle > start_angle else -1
    for angle in range(start_angle, int(target_angle) + step, step):
        set_servo_angle_instant(channel, angle)
        time.sleep(speed)

# =============================
# Calibration Mode
# =============================
def calibrate_joint(channel_name, channel_id):
    print(f"\n--- Calibrating {channel_name} (Channel {channel_id}) ---")
    print("Use 'a' to decrease, 'd' to increase, 's' to save/exit")
    
    angle = current_angles[channel_id]
    while True:
        set_servo_angle_instant(channel_id, angle)
        print(f"Current Angle: {angle}", end="\r")
        
        key = input("Step [a/d/s]: ").lower()
        if key == 'a': angle -= 2
        elif key == 'd': angle += 2
        elif key == 's': 
            print(f"\nSaved {channel_name} at {angle} degrees.")
            break

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    try:
        # Move to neutral first
        for i in range(6): set_servo_angle_instant(i, 90)
        
        print("1. Run Slow Test Sequence")
        print("2. Calibrate a Joint (Find safe limits)")
        choice = input("Select: ")

        if choice == '1':
            # Example: Move shoulder slowly
            move_servo_slow(SHOULDER, 130, speed=0.05)
            move_servo_slow(SHOULDER, 90, speed=0.05)
        elif choice == '2':
            chan = int(input("Which channel to calibrate (0-5)? "))
            calibrate_joint("TEST_JOINT", chan)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        pca.deinit()
