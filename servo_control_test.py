import serial
import time

# Make sure this matches your device
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # wait for KB2040 to initialize

def move_servo(angle):
    ser.write(f"{angle}\n".encode())  # send angle
    time.sleep(0.1)
    if ser.in_waiting:
        print(ser.readline().decode().strip())  # read KB2040 response

# Test sweep
try:
    while True:
        for angle in [0, 45, 90, 135, 180]:
            print(f"Moving to {angle}")
            move_servo(angle)
            time.sleep(1)

except KeyboardInterrupt:
    ser.close()
