import RPi.GPIO as GPIO
import time

SERVO_PIN = 18  # GPIO17 = physical pin 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
pwm.start(0)

def angle_to_duty(angle):
    # Map 0–180° to 2.5–12.5% duty cycle
    return 2.5 + (angle / 180.0) * 10.0

try:
    start = time.time()
    while time.time() - start < 15:   # run for 30 seconds
        # Sweep forward
        for ang in range(0, 181, 2):
            pwm.ChangeDutyCycle(angle_to_duty(ang))
            time.sleep(0.05)
        # Sweep backward
        for ang in range(180, -1, -2):
            pwm.ChangeDutyCycle(angle_to_duty(ang))
            time.sleep(0.05)

    pwm.ChangeDutyCycle(0)  # stop sending pulses
finally:
    pwm.stop()
    GPIO.cleanup()
