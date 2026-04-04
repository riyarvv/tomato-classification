import cv2
import serial
import time

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyUSB0', 115200)   # change if needed
time.sleep(2)

# ================= ARUCO =================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# ================= SETTINGS =================
TURN_THRESHOLD = 50
STOP_AREA = 12000

last_command = ""   # 🔥 prevents jitter

# ================= SEND =================
def send(cmd):
    global last_command

    # 🔥 only send if command changed
    if cmd != last_command:
        ser.write((cmd + "\n").encode())
        print("Sent:", cmd)
        last_command = cmd

# ================= MAIN LOOP =================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_center = w // 2

    # draw center line
    cv2.line(frame, (frame_center, 0), (frame_center, h), (0,255,255), 2)

    # detect marker
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:

        c = corners[0][0]

        # center of marker
        cx = int(c[:,0].mean())
        cy = int(c[:,1].mean())

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.circle(frame, (cx, cy), 6, (255,0,0), -1)

        # ================= CALCULATE =================
        error = cx - frame_center
        area = cv2.contourArea(c)

        # display info
        cv2.putText(frame, f"Error: {error}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Area: {int(area)}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # ================= CONTROL =================
        if abs(error) > TURN_THRESHOLD:

            if error > 0:
                send("R")   # RIGHT
            else:
                send("L")   # LEFT

        elif area < STOP_AREA:
            send("F")       # FORWARD

        else:
            send("S")       # STOP (reached)

    else:
        send("S")  # no marker → stop

    cv2.imshow("Aruco Navigation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
