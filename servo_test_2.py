from gpiozero import Servo
from time import sleep

servo = Servo(18, min_pulse_width=0.45/1000, max_pulse_width=2.45/1000)

try:
    while True:
        print("move to min")
        servo.min()
        sleep(1)
        print("move to max")
        servo.max()
        sleep(1)
except KeyboardInterrupt:
    servo.close()


