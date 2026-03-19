import RPi.GPIO as GPIO
import time

TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
time.sleep(2)

def get_distance():

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()

    # wait for echo start
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - start_time > 0.02:
            return -1   # timeout

    # wait for echo end
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - pulse_start > 0.02:
            return -1   # timeout

    duration = pulse_end - pulse_start
    distance = duration * 17150

    return distance

def get_stable_distance():
    readings = []
    for _ in range(5):
        readings.append(get_distance())
        time.sleep(0.05)
    return sum(readings)/len(readings)

try:
    while True:
        print("Distance:", get_stable_distance())
        time.sleep(0.5)

except KeyboardInterrupt:
    GPIO.cleanup()
