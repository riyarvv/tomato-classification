import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

# ============================
# PARAMETERS
# ============================

SERVO_CHANNELS = 6
MIN_PULSE = 500
MAX_PULSE = 2500
ACTUATION_RANGE = 180

# Global speed control (increase = slower)
SERVO_SPEED = 0.05   # ✅ SAFE & SLOW (fruit-friendly)

# ============================
# SAFE ANGLE CONFIGURATION
# ============================

SERVO_SAFE = {
    0: {"min": 20, "max": 160, "home": 90},   # Base
    1: {"min": 30, "max": 140, "home": 90},   # Shoulder
    2: {"min": 40, "max": 150, "home": 100},  # Elbow
    3: {"min": 20, "max": 160, "home": 90},   # Wrist
    4: {"min": 20, "max": 160, "home": 90},   # Extra axis
    5: {"min": 60, "max": 120, "home": 80},   # Gripper
}

# ============================
# INITIALIZATION
# ============================

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

servos = []
for i in range(SERVO_CHANNELS):
    servos.append(
        servo.Servo(
            pca.channels[i],
            actuation_range=ACTUATION_RANGE,
            min_pulse=MIN_PULSE,
            max_pulse=MAX_PULSE
        )
    )

# Track current servo positions
current_angles = {ch: SERVO_SAFE[ch]["home"] for ch in SERVO_SAFE}

# ============================
# SAFETY FUNCTIONS
# ============================

def clamp_angle(channel, angle):
    cfg = SERVO_SAFE[channel]
    return max(cfg["min"], min(cfg["max"], angle))


def move_servo_safe(channel, target_angle, speed=SERVO_SPEED):
    target_angle = clamp_angle(channel, target_angle)
    start_angle = current_angles[channel]

    step = 1 if target_angle > start_angle else -1

    print(f"Moving servo {channel} slowly → {target_angle}°")

    for angle in range(start_angle, target_angle + step, step):
        servos[channel].angle = angle
        time.sleep(speed)

    current_angles[channel] = target_angle


def move_to_home():
    print("\nMoving all servos slowly to HOME positions...")
    for ch, cfg in SERVO_SAFE.items():
        move_servo_safe(ch, cfg["home"])
    print("All servos at HOME.\n")

# ============================
# TEST MODES
# ============================

def manual_control(channel):
    print(f"\n--- MANUAL CONTROL (Servo {channel}) ---")
    print(f"Safe Range: {SERVO_SAFE[channel]['min']}° – {SERVO_SAFE[channel]['max']}°")
    print('Enter angle or "x" to return.\n')

    while True:
        choice = input("Angle: ")
        if choice.lower() == 'x':
            break
        try:
            angle = int(choice)
            move_servo_safe(channel, angle)
        except ValueError:
            print("Enter a valid number.")


def automatic_test(channel):
    print(f"\n--- AUTO TEST (Servo {channel}) ---")
    cfg = SERVO_SAFE[channel]

    test_angles = [
        cfg["min"],
        cfg["home"],
        cfg["max"],
        cfg["home"],
        cfg["min"]
    ]

    for a in test_angles:
        move_servo_safe(channel, a)
        time.sleep(0.5)

    print("Auto test complete.\n")

# ============================
# MAIN PROGRAM
# ============================

if __name__ == "__main__":
    try:
        move_to_home()

        while True:
            print("=" * 35)
            print("  6-AXIS ROBOTIC ARM TEST MENU")
            print("=" * 35)

            chan_input = input(
                f"Select channel (0–{SERVO_CHANNELS-1}) or 'q' to quit: "
            )

            if chan_input.lower() == 'q':
                break

            try:
                channel = int(chan_input)
                if channel not in SERVO_SAFE:
                    print("Invalid channel.")
                    continue
            except ValueError:
                print("Enter a valid number.")
                continue

            mode = input(
                "\n[1] Manual Safe (Slow)\n"
                "[2] Automatic Safe Sweep (Slow)\n"
                "Choice: "
            )

            if mode == '1':
                manual_control(channel)
            elif mode == '2':
                automatic_test(channel)
            else:
                print("Invalid selection.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    finally:
        print("\nShutting down PCA9685 safely.")
        pca.deinit()
