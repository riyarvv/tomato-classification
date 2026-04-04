import cv2
import serial
import time

# ================= SERIAL =================
ser = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)

# ================= ARUCO =================
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================= PARAMETERS =================
CENTER_TOLERANCE = 100   # bigger = more forward movement
HARD_TURN = 180          # only turn if very off
STOP_AREA = 40000

last_command = ""
last_time = time.time()

# ================= SEND FUNCTION =================
def send(cmd):
    global last_command, last_time

    # prevent spamming commands too fast
    if cmd != last_command and (time.time() - last_time > 0.1):
        ser.write((cmd + "\n").encode())
        print("Sent:", cmd)
        last_command = cmd
        last_time = time.time()

# ================= MAIN LOOP =================
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
        cy = int(c[:, 1].mean())

        area = cv2.contourArea(c)
        error = cx - frame_center

        print("Error:", error, "Area:", area)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)

        cv2.putText(frame, f"Err:{error}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Area:{int(area)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ================= CONTROL LOGIC =================
        if area > STOP_AREA:
            send("S")

        else:
            # prioritize forward
            if abs(error) < CENTER_TOLERANCE:
                send("F")

            # only turn if clearly off
            elif error < -HARD_TURN:
                send("L")

            elif error > HARD_TURN:
                send("R")

            else:
                # small correction → still move forward
                send("F")

    else:
        send("S")

    cv2.imshow("Aruco Navigation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
