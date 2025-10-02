import RPi.GPIO as GPIO
import time

SERVO_PIN = 17  # GPIO17 = physical pin 11

GPIO.setmode(GPIO.BCM)        # use BCM numbering
GPIO.setup(SERVO_PIN, GPIO.OUT)

# 50 Hz PWM (20ms period, typical for servos)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def angle_to_duty(angle):
    # Map 0–180° to duty cycle (2.5–12.5% works for most micro servos)
    return 2.5 + (angle / 180.0) * 10.0

try:
    start_time = time.time()
    while time.time() - start_time < 10:  # run for 10 seconds
        # sweep servo back and forth
        for ang in [0, 90, 180, 90]:
            pwm.ChangeDutyCycle(angle_to_duty(ang))
            time.sleep(0.5)  # wait between moves
    pwm.ChangeDutyCycle(0)  # stop sending pulses

finally:
    pwm.stop()
    GPIO.cleanup()
