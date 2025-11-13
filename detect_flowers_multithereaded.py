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
import threading

# --------------- Config ---------------
PROTOTXT = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"

VIDEO_DEVICE = 0                      # camera index
USE_V4L2_backend = True               # set True on many Pi setups
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N = 3

CONF_THRESHOLD = 0.7                  # detection confidence threshold to consider
LED_PIN = 17                          # BCM pin, physical pin 11
INDICATOR_LED_PIN = 27                # BCM pin
SERVO_PIN = 18                        # BCM pin (GPIO18 = physical pin 12)

# Duration servo/LED should run when triggered
SERVO_RUN_SECONDS = 5.0             # run for 5 seconds when triggered

TARGET_CLASS = "pottedplant"          # class that triggers LED+servo

# ---------------- Helper functions ----------------
def kill_camera_processes():
    """Kill any processes holding /dev/video0."""
    try:
        result = subprocess.run(["sudo", "lsof", "/dev/video0"],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return
        for line in lines[1:]:
            parts = line.split()
            if len(parts) > 1:
                pid = int(parts[1])
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass

def reset_gpio():
    """Try to safely clear GPIO state (best-effort)."""
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
    except Exception:
        pass

def reload_camera_driver():
    """Reload the bcm2835-v4l2 driver (for CSI camera)."""
    try:
        subprocess.run(["sudo", "modprobe", "-r", "bcm2835_v4l2"], check=False)
        time.sleep(0.3)
        subprocess.run(["sudo", "modprobe", "bcm2835_v4l2"], check=False)
    except Exception:
        pass

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
        try:
            if use_v4l2:
                cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(device_index)
        except Exception:
            cap = cv2.VideoCapture(device_index)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            time.sleep(0.3)
            return cap
        else:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            time.sleep(delay)
    return None

# ---------------- Threading globals ----------------
servo_event = threading.Event()   # set by detection thread to request servo run
stop_event = threading.Event()    # set to signal threads to stop
pwm = None                        # PWM instance (created once)
pwm_lock = threading.Lock()       # protect pwm in multi-threaded access

# ---------------- Worker threads ----------------
def servo_worker():
    """Wait for servo_event; when set, perform LED+servo activity for SERVO_RUN_SECONDS.
    Retriggers during an active run are ignored."""
    global pwm
    # Initialize PWM once
    with pwm_lock:
        if pwm is None:
            try:
                pwm = GPIO.PWM(SERVO_PIN, 50)
                pwm.start(0)
            except Exception as e:
                print("Servo PWM init failed:", e)
                pwm = None

    busy_until = 0.0
    while not stop_event.is_set():
        # wait for trigger up to short timeout so we can check stop_event regularly
        triggered = servo_event.wait(timeout=0.2)
        if stop_event.is_set():
            break
        if not triggered:
            continue
        # clear the event immediately
        servo_event.clear()
        now = time.time()
        # if already running, ignore retrigger
        if now < busy_until:
            # already handling; ignore this trigger
            continue
        busy_until = now + SERVO_RUN_SECONDS

        # turn ON indicator LED (and main LED)
        try:
            GPIO.output(INDICATOR_LED_PIN, GPIO.HIGH)
            GPIO.output(LED_PIN, GPIO.HIGH)
        except Exception:
            pass

        # Run servo sweep for SERVO_RUN_SECONDS (coarse steps to be faster)
        sweep_start = time.time()
        try:
            while time.time() - sweep_start < SERVO_RUN_SECONDS and not stop_event.is_set():
                # forward
                for ang in range(0, 181, 6):
                    if stop_event.is_set() or time.time() - sweep_start >= SERVO_RUN_SECONDS:
                        break
                    if pwm is not None:
                        try:
                            pwm.ChangeDutyCycle(angle_to_duty(ang))
                        except Exception:
                            pass
                    time.sleep(0.03)
                # backward
                for ang in range(180, -1, -6):
                    if stop_event.is_set() or time.time() - sweep_start >= SERVO_RUN_SECONDS:
                        break
                    if pwm is not None:
                        try:
                            pwm.ChangeDutyCycle(angle_to_duty(ang))
                        except Exception:
                            pass
                    time.sleep(0.03)
        finally:
            # stop sending PWM pulses (keep servo safe)
            try:
                if pwm is not None:
                    pwm.ChangeDutyCycle(0)
            except Exception:
                pass
            # turn off LEDs
            try:
                GPIO.output(LED_PIN, GPIO.LOW)
                GPIO.output(INDICATOR_LED_PIN, GPIO.LOW)
            except Exception:
                pass

    # cleanup PWM when stopping
    with pwm_lock:
        if pwm is not None:
            try:
                pwm.stop()
            except Exception:
                pass

def detection_worker(cap, net, classes):
    """Detection that only runs inference on every Nth frame to reduce CPU load."""
    process_counter = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        process_counter += 1
        if PROCESS_EVERY_N > 0 and (process_counter % PROCESS_EVERY_N) != 0:
            # skip this frame (cheap)
            continue

        # Do inference on this frame
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(inp, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        try:
            detections = net.forward()
        except Exception as e:
            print("Inference error:", e)
            continue

        # scan detections (same as before)
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < CONF_THRESHOLD:
                continue
            idx = int(detections[0, 0, i, 1])
            if idx < 0 or idx >= len(classes):
                continue
            label = classes[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            print(f"Detected {label} ({conf:.2f}) at ({centerX},{centerY})")
            if label == TARGET_CLASS:
                servo_event.set()
        # tiny sleep to yield CPU
        time.sleep(0.001)

# ---------------- Main ----------------
def main():
    kill_camera_processes()
    reset_gpio()
    reload_camera_driver()
    print("Starting detection (press 'q' in window to stop if you have a display).")

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    try:
        setup_gpio()

        net = load_model(PROTOTXT, MODEL)
    except Exception as e:
        print("Startup error:", e)
        traceback.print_exc()
        return

    cap = open_camera(VIDEO_DEVICE, use_v4l2=USE_V4L2_backend, retries=6, delay=0.7)
    if cap is None:
        print("ERROR: camera not available.")
        return

    # Flicker indicator LED quickly to show program started
    for _ in range(3):
        try:
            GPIO.output(INDICATOR_LED_PIN, GPIO.HIGH)
            time.sleep(0.15)
            GPIO.output(INDICATOR_LED_PIN, GPIO.LOW)
            time.sleep(0.15)
        except Exception:
            pass

    # Start servo thread
    sv_thread = threading.Thread(target=servo_worker, daemon=True)
    sv_thread.start()

    # Start detection in its own thread (so main can wait / join or handle UI)
    det_thread = threading.Thread(target=detection_worker, args=(cap, net, CLASSES), daemon=True)
    det_thread.start()

    try:
        # main loop: wait until stop_event is set (threads will run)
        while not stop_event.is_set():
            # you can add lightweight monitoring here if desired
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        # signal threads to stop and wait briefly
        stop_event.set()
        time.sleep(0.3)

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # safe GPIO cleanup only if a mode is set
        try:
            mode = GPIO.getmode()
        except Exception:
            mode = None
        if mode is not None:
            try:
                # ensure LEDs off
                GPIO.output(LED_PIN, GPIO.LOW)
                GPIO.output(INDICATOR_LED_PIN, GPIO.LOW)
            except Exception:
                pass
            try:
                GPIO.cleanup()
            except Exception:
                pass

        print("Stopped and cleaned up.")

if __name__ == "__main__":
    main()
