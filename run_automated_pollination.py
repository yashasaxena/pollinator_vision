#!/usr/bin/env python3
import subprocess
import numpy as np
import os
import signal
import time
import threading
import RPi.GPIO as GPIO
from ultralytics import YOLO

# ---------------- Config ----------------
WIDTH, HEIGHT = 640, 480
MODEL_PATH = "best_ncnn_model/"   # ← only change from previous code
CLASSES = ["Bulb", "Ring", "Waffle"]
TRIGGER_CLASSES = {"Bulb", "Ring", "Waffle"}

# ================================================================
# TUNING PARAMETERS
# ================================================================
CONF_THRESHOLD     = 0.3
PERSIST_FRAMES     = 2
SERVO_HOLD_SECONDS = 2.0
COOLDOWN_SECONDS   = 1.0
# ================================================================

# ---------------- Servo setup ----------------
SERVO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

def angle_to_duty(angle):
    return 2.5 + (angle / 180.0) * 10.0

pwm.ChangeDutyCycle(angle_to_duty(0))
time.sleep(0.3)
pwm.ChangeDutyCycle(0)

servo_lock          = threading.Lock()
servo_active        = False
last_servo_end_time = 0.0

def trigger_servo():
    global servo_active, last_servo_end_time
    with servo_lock:
        if servo_active:
            return
        if time.time() - last_servo_end_time < COOLDOWN_SECONDS:
            print("  [servo] cooldown, skipping")
            return
        servo_active = True
    try:
        print(f"  [servo] ON  {time.time():.3f}")
        pwm.ChangeDutyCycle(angle_to_duty(90))
        time.sleep(SERVO_HOLD_SECONDS)
        pwm.ChangeDutyCycle(angle_to_duty(0))
        time.sleep(0.3)
        pwm.ChangeDutyCycle(0)
    finally:
        with servo_lock:
            servo_active = False
            last_servo_end_time = time.time()
        print(f"  [servo] OFF {time.time():.3f}")

# ---------------- Kill leftovers ----------------
subprocess.run(["pkill", "-f", "rpicam-vid"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "ffmpeg"], stderr=subprocess.DEVNULL)
time.sleep(0.5)

# ---------------- Load NCNN model ----------------
print("Loading NCNN model...")
model = YOLO(MODEL_PATH)
print("Model ready")

# ---------------- Camera pipeline ----------------
cam_cmd = [
    "rpicam-vid",
    "--timeout", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", "30",
    "--inline",
    "--codec", "h264",
    "--profile", "baseline",
    "--flush",
    "--nopreview",
    "-o", "-"
]

ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "quiet",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-f", "h264",
    "-i", "pipe:0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

print("Starting camera...")
p1 = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, preexec_fn=os.setsid)
p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, preexec_fn=os.setsid)
p1.stdout.close()

frame_size             = WIDTH * HEIGHT * 3
frame_count            = 0
consecutive_detections = 0

# ---------------- Main loop ----------------
try:
    while True:
        raw = p2.stdout.read(frame_size)
        if not raw or len(raw) != frame_size:
            print("Pipeline ended")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        frame_count += 1

        if frame_count % 3 != 0:
            continue

        # Subsample ~320x240 without cv2
        frame_small = frame[::2, ::2]

        results = model(frame_small, conf=CONF_THRESHOLD, imgsz=320, verbose=False)

        target_seen = False
        for r in results:
            for box in r.boxes:
                cls        = int(box.cls[0])
                conf       = float(box.conf[0])
                class_name = CLASSES[cls]
                print(f"  {class_name}: {conf:.2f}")
                if class_name in TRIGGER_CLASSES:
                    target_seen = True

        if target_seen:
            consecutive_detections += 1
            print(f"  streak: {consecutive_detections}/{PERSIST_FRAMES}")
        else:
            consecutive_detections = 0

        if consecutive_detections >= PERSIST_FRAMES:
            threading.Thread(target=trigger_servo, daemon=True).start()
            consecutive_detections = 0

        if len(results) > 0:
            print("----")

except KeyboardInterrupt:
    print("Stopping (Ctrl+C)...")

finally:
    print("Cleaning up...")
    pwm.stop()
    GPIO.cleanup()
    for p in (p2, p1):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    for p in (p2, p1):
        try:
            p.wait(timeout=2)
        except Exception:
            pass
    print("Done.")
