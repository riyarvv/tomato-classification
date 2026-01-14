import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# ----------------------------
# PARAMETERS
# ----------------------------
SERVO_CHANNELS = 6         # Total servos in your arm
MIN_PULSE = 500            # Standard min pulse (usually 500-750)
MAX_PULSE = 2500           # Standard max pulse (usually 2250-2500)
MAX_ANGLE = 180            # Adjusted to standard 180 degrees

# ----------------------------
# INITIALIZATION
# ----------------------------
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Create a list to hold all 6 servo objects
servos = []
for i in range(SERVO_CHANNELS):
    # actuation_range: the physical degrees of the servo
    # min_pulse/max_pulse: calibration for the PWM signal
    servos.append(servo.Servo(pca.channels[i], actuation_range=MAX_ANGLE, 
                              min_pulse=MIN_PULSE, max_pulse=MAX_PULSE))

# ----------------------------
# FUNCTIONS
# ----------------------------

def move_servo(channel, angle):
    """Moves a specific servo to the desired angle."""
    if 0 <= angle <= MAX_ANGLE:
        print(f"Moving servo on channel {channel} to {angle} degrees")
        servos[channel].angle = angle
    else:
        print(f"Error: Angle {angle} is out of range (0-{MAX_ANGLE})")

def manual_control(channel):
    print(f"\n--- Manual Control: Channel {channel} ---")
    print('Enter angle (0-180) or "x" to return to main menu.')
    while True:
        choice = input("Angle: ")
        if choice.lower() == 'x':
            break
        try:
            angle = int(choice)
            move_servo(channel, angle)
        except ValueError:
            print("Invalid input. Please enter a number.")

def automatic_test(channel):
    print(f"\n--- Running Auto Test on Channel {channel} ---")
    test_angles = [0, 90, 180, 90, 0]
    for a in test_angles:
        move_servo(channel, a)
        time.sleep(1)
    print("Test complete.")

# ----------------------------
# MAIN LOOP
# ----------------------------
if __name__ == "__main__":
    try:
        while True:
            print("\n" + "="*30)
            print("6-AXIS ROBOTIC ARM CONTROLLER")
            print("="*30)
            
            chan_input = input(f"Which channel (0-{SERVO_CHANNELS-1})? (or 'q' to quit): ")
            if chan_input.lower() == 'q':
                break
                
            try:
                channel = int(chan_input)
                if not (0 <= channel < SERVO_CHANNELS):
                    print(f"Invalid channel. Choose 0 to {SERVO_CHANNELS-1}.")
                    continue
            except ValueError:
                print("Please enter a valid channel number.")
                continue

            mode = input("[1] Manual Test\n[2] Automatic Sweep\nSelection: ")
            
            if mode == '1':
                manual_control(channel)
            elif mode == '2':
                automatic_test(channel)
            else:
                print("Invalid selection.")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    finally:
        pca.deinit() # Cleanly shut down the PCA9685
