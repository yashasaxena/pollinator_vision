#!/usr/bin/env python3
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import sys
import traceback
from pathlib import Path
import os
import signal
import subprocess
import time

# --------------- Config ---------------
PROTOTXT = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"

VIDEO_DEVICE = 0                      # camera index
USE_V4L2_backend = True               # set True on many Pi setups
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONF_THRESHOLD = 0.7                  # detection confidence threshold to consider
LED_PIN = 17                          # BCM pin, physical pin 11
INDICATOR_LED_PIN = 27                # physical pin 
SERVO_PIN = 18  # GPIO18 = physical pin 12
LED_ON_DURATION = 5.0                 # seconds to keep LED on after detection
TARGET_CLASS = "pottedplant"          # class that triggers LED

# --------------- Helper functions ---------------
def kill_camera_processes():
    """Kill any processes holding /dev/video0."""
    print("🔍 Checking for processes using /dev/video0 ...")
    try:
        result = subprocess.run(
            ["sudo", "lsof", "/dev/video0"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            print("✅ No active camera processes found.")
            return

        for line in lines[1:]:
            parts = line.split()
            if len(parts) > 1:
                pid = int(parts[1])
                print(f"❌ Killing process PID {pid}: {parts[0]}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print(f"⚠️ Could not kill PID {pid}: {e}")
    except Exception as e:
        print(f"Error checking camera processes: {e}")

def reset_gpio():
    """Turn off LED and reset GPIO state."""
    print("🧹 Cleaning up GPIO...")
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("✅ GPIO cleaned up successfully.")
    except Exception as e:
        print(f"⚠️ GPIO cleanup error: {e}")

def reload_camera_driver():
    """Reload the bcm2835-v4l2 driver (for CSI camera)."""
    print("🔁 Reloading bcm2835-v4l2 driver (if loaded)...")
    try:
        subprocess.run(["sudo", "modprobe", "-r", "bcm2835_v4l2"], check=False)
        time.sleep(0.5)
        subprocess.run(["sudo", "modprobe", "bcm2835_v4l2"], check=False)
        print("✅ Driver reload attempted.")
    except Exception as e:
        print(f"⚠️ Driver reload failed: {e}")

def setup_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    GPIO.setup(INDICATOR_LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.output(INDICATOR_LED_PIN, GPIO.LOW)
    

def angle_to_duty(angle):
    # Map 0–180° to 2.5–12.5% duty cycle
    return 2.5 + (angle / 180.0) * 10.0

def load_model(prototxt_path: str, model_path: str):
    p1 = Path(prototxt_path)
    p2 = Path(model_path)
    if not p1.exists() or not p2.exists():
        raise FileNotFoundError(f"Model files not found: {p1.resolve()}, {p2.resolve()}")
    net = cv2.dnn.readNetFromCaffe(str(p1), str(p2))
    return net

def open_camera(device_index=0, use_v4l2=True, retries=5, delay=0.6):
    for attempt in range(retries):
        if use_v4l2:
            cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(device_index)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            # warm up camera a little
            time.sleep(0.5)
            return cap
        else:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            time.sleep(delay)
    return None

# --------------- Main ---------------
def main():
    kill_camera_processes()
    reset_gpio()
    reload_camera_driver()
    print("✅ Reset complete. Try rerunning your detection script now.")
    # Load labels
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    last_detection_time = 0.0

    try:
        print("Setting up GPIO...")
        setup_gpio()

        print("Loading model...")
        net = load_model(PROTOTXT, MODEL)
        print("Model loaded.")

        print("Opening camera...")
        cap = open_camera(VIDEO_DEVICE, use_v4l2=USE_V4L2_backend, retries=6, delay=0.7)
        if cap is None:
            print("ERROR: Camera not found or could not be opened. Make sure camera is enabled and connected.")
            return

        print("Camera opened. Starting main loop. Press 'q' to quit.")
        # LED flicker to indicate detections have started
        for i in range(3):
            GPIO.output(INDICATOR_LED_PIN, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(INDICATOR_LED_PIN, GPIO.LOW)
            time.sleep(0.2)
        print("LED flicker complete — detection running.")
        while True:
            ret, frame = cap.read()
            if not ret:
                # camera lost frame, try a short recovery
                print("Warning: failed to read frame from camera; attempting to continue.")
                time.sleep(0.1)
                continue

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            found_target = False

            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < CONF_THRESHOLD:
                    continue

                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx >= len(CLASSES):
                    continue

                label = CLASSES[idx]
                if label == TARGET_CLASS:
                    # found_target = True

                    # compute bounding box if you want to draw / log
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    centerX = int((startX + endX) / 2)
                    centerY = int((startY + endY) / 2)
                    print(f"Detected {label} ({confidence:.2f}) at ({centerX}, {centerY})")
                    # we don't break so we can see if multiple matches happen; optional break here

                    GPIO.output(LED_PIN, GPIO.HIGH)
                    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
                    pwm.start(0)

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
                        
                else:
                    GPIO.output(LED_PIN, GPIO.LOW)
                
            # GPIO.cleanup()

            # Manage LED without blocking the camera loop:
            # if found_target:
            #     last_detection_time = time.time()

            # # If last detection within LED_ON_DURATION, keep LED on
            # if (time.time() - last_detection_time) <= LED_ON_DURATION:
            #     GPIO.output(LED_PIN, GPIO.HIGH)
            #     # optional: only print on transitions; kept simple here
            #     print("LED ON")
            # else:
            #     GPIO.output(LED_PIN, GPIO.LOW)
            #     print("LED Off")

            # Optional: show frame with a small overlay (uncomment if running with GUI/display)
            # cv2.imshow("MobileNet SSD Detection", frame)

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            GPIO.output(LED_PIN, GPIO.LOW)
        except Exception:
            pass
        GPIO.cleanup()
        print("Cleaned up and exiting.")

if __name__ == "__main__":
    main()
