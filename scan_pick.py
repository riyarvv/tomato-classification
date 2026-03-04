import time
import cv2
from adafruit_servokit import ServoKit

# ==============================
# SERVO SETUP
# ==============================

kit = ServoKit(channels=16)

# Servo Channels (Change if different)
BASE_CH = 0
SHOULDER_CH = 1
ELBOW_CH = 2
GRIPPER_CH = 3
CAMERA_CH = 4

servos = kit.servo

# ==============================
# SERVO LIMITS (SAFE)
# ==============================

LIMITS = {
    BASE_CH: {"neutral": 20, "min": 10, "max": 100},

    SHOULDER_CH: {
        "neutral": 90,
        "min": 20,
        "max": 140,
        "pick": 120
    },

    ELBOW_CH: {
        "neutral": 90,
        "min": 30,
        "max": 150,
        "pick": 110
    },

    GRIPPER_CH: {
        "neutral": 15,   # fully open
        "min": 10,
        "max": 100
    }
}

# ==============================
# SMOOTH MOVEMENT FUNCTION
# ==============================

def move_smooth(channel, target, step=1, delay=0.03):

    target = max(LIMITS[channel]["min"],
                 min(target, LIMITS[channel]["max"]))

    current = servos[channel].angle
    if current is None:
        current = target

    current = int(current)
    target = int(target)

    if current < target:
        angles = range(current, target + 1, step)
    else:
        angles = range(current, target - 1, -step)

    for angle in angles:
        angle = max(LIMITS[channel]["min"],
                    min(angle, LIMITS[channel]["max"]))
        servos[channel].angle = angle
        time.sleep(delay)


# ==============================
# GRIPPER FUNCTIONS
# ==============================

def gripper_close_slow():
    steps = [15, 30, 45, 60, 75, 90, 100]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)


def gripper_open_slow():
    steps = [100, 90, 75, 60, 45, 30, 15]
    for angle in steps:
        servos[GRIPPER_CH].angle = angle
        time.sleep(1.5)


# ==============================
# PICK FUNCTION
# ==============================

def pick_and_drop():

    print("🍅 Tomato Detected - Starting Pick")

    # Move base to pick position (20°)
    move_smooth(BASE_CH, LIMITS[BASE_CH]["neutral"])

    # Move arm towards tomato (gripper OPEN)
    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["pick"])
    move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["pick"])

    time.sleep(1)

    # CLOSE gripper ONLY NOW
    gripper_close_slow()

    time.sleep(1)

    # Lift arm back up
    move_smooth(ELBOW_CH, LIMITS[ELBOW_CH]["neutral"])
    move_smooth(SHOULDER_CH, LIMITS[SHOULDER_CH]["neutral"])

    time.sleep(1)

    # DROP at same base position
    gripper_open_slow()

    print("✅ Pick Completed")


# ==============================
# INITIAL POSITION
# ==============================

servos[BASE_CH].angle = LIMITS[BASE_CH]["neutral"]
servos[SHOULDER_CH].angle = LIMITS[SHOULDER_CH]["neutral"]
servos[ELBOW_CH].angle = LIMITS[ELBOW_CH]["neutral"]
servos[GRIPPER_CH].angle = LIMITS[GRIPPER_CH]["neutral"]
servos[CAMERA_CH].angle = LIMITS[BASE_CH]["neutral"]

time.sleep(2)

# ==============================
# CAMERA SETUP
# ==============================

cap = cv2.VideoCapture(0)

# ==============================
# SCANNING VARIABLES
# ==============================

scan_angle = LIMITS[BASE_CH]["neutral"]
scan_direction = 1

print("🤖 Starting Automatic Harvest Mode")

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ---- YOUR YOLO DETECTION LOGIC GOES HERE ----
    # For now, we simulate detection
    tomato_detected = False

    # Example dummy condition:
    # tomato_detected = True  # Uncomment to test pick

    if tomato_detected:
        pick_and_drop()

    else:
        # SCANNING MODE
        scan_angle += scan_direction * 2

        if scan_angle >= LIMITS[BASE_CH]["max"]:
            scan_direction = -1

        if scan_angle <= LIMITS[BASE_CH]["min"]:
            scan_direction = 1

        move_smooth(BASE_CH, scan_angle, step=1, delay=0.02)
        servos[CAMERA_CH].angle = servos[BASE_CH].angle

        time.sleep(0.05)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
