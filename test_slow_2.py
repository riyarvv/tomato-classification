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

# Setup 6 servos using the proper library helper
# This library automatically handles the 16-bit duty cycle math
servos = []
for i in range(6):
    # Adjust min_pulse and max_pulse if your servos don't reach full range
    s = servo.Servo(pca.channels[i], min_pulse=500, max_pulse=2500)
    servos.append(s)

# =============================
# SLOW MOVEMENT FUNCTION
# =============================
def move_slow(channel_id, target_angle, speed=0.05):
    """
    Moves a servo slowly.
    speed: delay in seconds between each degree.
    """
    # Get the starting angle (default to 90 if unknown)
    current = servos[channel_id].angle
    if current is None: 
        current = 90
        servos[channel_id].angle = 90
    
    start_angle = int(current)
    target_angle = int(max(0, min(180, target_angle))) # Clamp to safety
    
    if start_angle == target_angle:
        return

    step = 1 if target_angle > start_angle else -1
    
    print(f"Moving Channel {channel_id} to {target_angle}...")
    
    for angle in range(start_angle, target_angle + step, step):
        servos[channel_id].angle = angle
        time.sleep(speed)

# =============================
# MAIN TEST
# =============================
if __name__ == "__main__":
    try:
        print("Homing all servos to 90 degrees...")
        for i in range(6):
            servos[i].angle = 90
            time.sleep(0.1) # Small stagger to avoid power surge
            
        time.sleep(1)
        
        # Example Test: Move the Base (0) and Shoulder (1)
        move_slow(0, 130, speed=0.05)
        move_slow(0, 50, speed=0.05)
        move_slow(0, 90, speed=0.05)
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pca.deinit()
