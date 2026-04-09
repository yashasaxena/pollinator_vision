#!/usr/bin/env python3
import subprocess
import shutil
import numpy as np
import cv2
import pigpio
import time
import threading

WIDTH, HEIGHT, FPS = 640, 480, 30
DURATION_MS = 50000  # extended for longer use

# --- Servo config ---
SERVO_PIN = 18
SERVO_ON_PW  = 2000   # pulsewidth in µs → adjust for your servo's "on" position
SERVO_OFF_PW = 1000   # pulsewidth in µs → "off" / neutral position
SERVO_HOLD_SECONDS = 3  # how long to hold the servo after detecting a plant

pi = pigpio.pi()
if not pi.connected:
    raise SystemExit("pigpio daemon not running — run: sudo pigpiod")

# --- Servo state ---
servo_lock = threading.Lock()
servo_active = False  # True while the servo is currently being held on

def trigger_servo(hold_seconds=SERVO_HOLD_SECONDS):
    """
    Runs in a background thread.
    Moves the servo to ON position, holds it, then returns it to OFF.
    Ignores new triggers while already active.
    """
    global servo_active
    with servo_lock:
        if servo_active:
            return          # already running — skip duplicate trigger
        servo_active = True

    try:
        pi.set_servo_pulsewidth(SERVO_PIN, SERVO_ON_PW)
        time.sleep(hold_seconds)
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, SERVO_OFF_PW)
        with servo_lock:
            servo_active = False


def find_tool():
    for name in ("libcamera-vid", "raspivid", "rpicam-vid"):
        if shutil.which(name):
            return name
    return None

def build_cmd(tool):
    if tool == "libcamera-vid":
        return [tool, "--output", "-", "--timeout", str(DURATION_MS),
                "--width", str(WIDTH), "--height", str(HEIGHT),
                "--framerate", str(FPS)]
    if tool == "raspivid":
        return [tool, "-o", "-", "-t", str(DURATION_MS),
                "-w", str(WIDTH), "-h", str(HEIGHT), "-fps", str(FPS)]
    return [tool, "-o", "-", "-t", str(DURATION_MS)]


# Load MobileNet SSD
prototxt = "deploy.prototxt"
model    = "mobilenet_iter_73000.caffemodel"
net      = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

tool = find_tool()
if not tool:
    print("No capture tool found (libcamera-vid/raspivid/rpicam-vid).")
    raise SystemExit(1)

vid_cmd = build_cmd(tool)
ff_cmd  = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "h264", "-i", "pipe:0",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
]

print("Running:", " ".join(vid_cmd))
pi.set_servo_pulsewidth(SERVO_PIN, SERVO_OFF_PW)  # start in neutral

with subprocess.Popen(vid_cmd, stdout=subprocess.PIPE) as vid_proc, \
     subprocess.Popen(ff_cmd, stdin=vid_proc.stdout, stdout=subprocess.PIPE) as ff_proc:

    vid_proc.stdout.close()
    frame_size = WIDTH * HEIGHT * 3

    try:
        while True:
            raw = ff_proc.stdout.read(frame_size)
            if not raw:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
            )
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx   = int(detections[0, 0, i, 1])
                    box   = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
                    print(label)

                    # --- Potted plant trigger ---
                    if CLASSES[idx] == "pottedplant":
                        print("🌿 Potted plant detected — triggering servo")
                        t = threading.Thread(target=trigger_servo, daemon=True)
                        t.start()

    except KeyboardInterrupt:
        pass
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # release servo signal fully
        pi.stop()
        ff_proc.terminate()
        vid_proc.terminate()
        ff_proc.wait(timeout=1)
        vid_proc.wait(timeout=1)
