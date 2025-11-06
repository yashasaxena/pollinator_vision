import RPi.GPIO as GPIO
import time

# Use BCM numbering
GPIO.setmode(GPIO.BCM)

# Choose the GPIO pin (for example, GPIO17)
LED_PIN = 17

# Set the pin as an output
GPIO.setup(LED_PIN, GPIO.OUT)

# Turn on the LED
GPIO.output(LED_PIN, GPIO.HIGH)

print("LED is ON")

# Keep it on for 5 seconds
time.sleep(5)

# Turn off the LED
GPIO.output(LED_PIN, GPIO.LOW)
print("LED is OFF")

# Clean up the GPIO settings
GPIO.cleanup()
