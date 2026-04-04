import cv2
import serial
import time

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyACM0', 115200)  # Pi port
time.sleep(2)

# ================= ARUCO =================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()
else:
    print("✅ Camera opened")

TURN_THRESHOLD = 50
STOP_AREA = 12000

last_command = ""

def send(cmd):
    global last_command
    if cmd != last_command:
        ser.write((cmd + "\n").encode())
        print("Sent:", cmd)
        last_command = cmd

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_center = w // 2

    cv2.line(frame, (frame_center, 0), (frame_center, h), (0,255,255), 2)

    # 🔥 OLD VERSION COMPATIBLE
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)

    if ids is not None and len(corners) > 0:

        c = corners[0][0]

        cx = int(c[:,0].mean())
        cy = int(c[:,1].mean())

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.circle(frame, (cx, cy), 6, (255,0,0), -1)

        error = cx - frame_center
        area = cv2.contourArea(c)

        cv2.putText(frame, f"Error: {error}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if abs(error) > TURN_THRESHOLD:
            if error > 0:
                send("R")
            else:
                send("L")

        elif area < STOP_AREA:
            send("F")

        else:
            send("S")

    else:
        send("S")

    cv2.imshow("Aruco Navigation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
