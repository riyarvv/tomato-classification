import cv2
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

cap = cv2.VideoCapture(0)

TURN_THRESHOLD = 60
STOP_AREA = 40000

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

    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)

    if ids is not None and len(corners) > 0:
        c = corners[0][0]

        cx = int(c[:, 0].mean())
        area = cv2.contourArea(c)

        error = cx - frame_center

        print("Error:", error, "Area:", area)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.putText(frame, f"Err:{error}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Area:{int(area)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ✅ better logic
        if area > STOP_AREA:
            send("S")

        else:
            if error < -TURN_THRESHOLD:
                send("L")
            elif error > TURN_THRESHOLD:
                send("R")
            else:
                send("F")

    else:
        send("S")

    cv2.imshow("Aruco Navigation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
