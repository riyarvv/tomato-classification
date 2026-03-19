from gpiozero import DistanceSensor
from time import sleep

sensor = DistanceSensor(echo=24, trigger=23)

def get_distance():
    return sensor.distance * 100  # meters → cm

def get_stable_distance():
    readings = []
    for _ in range(5):
        readings.append(get_distance())
        sleep(0.05)
    return sum(readings) / len(readings)

try:
    while True:
        print(f"Distance: {get_stable_distance():.2f} cm")
        sleep(0.5)

except KeyboardInterrupt:
    print("Stopped")
