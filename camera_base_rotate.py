import time
from adafruit_servokit import ServoKit

# =====================================
# PCA9685 Initialization
# =====================================
# 16-channel PCA9685, default I2C address 0x40
kit = ServoKit(channels=16)

# =====================================
# Servo Calibration (important for SG90)
# =====================================
# Pulse width range in microseconds
kit.servo[0].set_pulse_width_range(500, 2500)

# =====================================
# Main Program
# =====================================
try:
    while True:
        # Move servo from 20° to 50°
        for angle in range(20, 51, 1):
            kit.servo[0].angle = angle
            time.sleep(0.03)   # controls speed

        time.sleep(1)  # hold at 50°

        # Move servo from 50° back to 20°
        for angle in range(50, 19, -1):
            kit.servo[0].angle = angle
            time.sleep(0.03)

        time.sleep(1)  # hold at 20°

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    # Release the servo (no PWM signal)
    kit.servo[0].angle = None
