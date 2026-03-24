import board
import pwmio
from adafruit_motor import servo
import time

pwm = pwmio.PWMOut(board.GP5, duty_cycle=2**15, frequency=50)

my_servo = servo.Servo(pwm)

while True:
    my_servo.angle = 0
    time.sleep(1)

    my_servo.angle = 90
    time.sleep(1)

    my_servo.angle = 180
    time.sleep(1)