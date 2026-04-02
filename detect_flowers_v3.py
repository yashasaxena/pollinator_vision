#!/usr/bin/env python3
import cv2
import numpy as np
import time
import traceback
from pathlib import Path
import subprocess, shutil
import threading

from gpiozero import Device, Servo
from gpiozero.pins.lgpio import LGPIOFactory

# -------- Force Pi5 GPIO backend --------
Device.pin_factory = LGPIOFactory()

# --------------- Config ---------------
PROTOTXT = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

PROCESS_EVERY_N = 5
CONF_THRESHOLD = 0.6
SERVO_RUN_SECONDS = 4.0

TARGET_CLASS = "pottedplant"

# ---------------- Camera pipeline ----------------
def find_tool():
    for name in ("rpicam-vid", "libcamera-vid", "raspivid"):
        if shutil.which(name):
            return name
    return None

def build_cmd(tool):
    return [
        tool,
        "--output", "-",
        "--timeout", "0",
        "--width", str(FRAME_WIDTH),
        "--height", str(FRAME_HEIGHT),
        "--framerate", str(FPS),
        "--codec", "h264",
        "--nopreview"
    ]

class CameraStream:
    def __init__(self):
        self.frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3

    def start(self):
        tool = find_tool()
        if not tool:
            raise RuntimeError("No camera tool found")

        vid_cmd = build_cmd(tool)
        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "h264", "-i", "pipe:0",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"
        ]

        self.vid = subprocess.Popen(vid_cmd, stdout=subprocess.PIPE)
        self.ff = subprocess.Popen(ff_cmd, stdin=self.vid.stdout, stdout=subprocess.PIPE)
        self.vid.stdout.close()

    def read(self):
        raw = self.ff.stdout.read(self.frame_size)
        if not raw:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        return True, frame

    def release(self):
        for p in (self.ff, self.vid):
            try:
                p.terminate()
            except:
                pass

# ---------------- Model ----------------
def load_model():
    return cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# ---------------- Servo ----------------
servo_event = threading.Event()
stop_event = threading.Event()

def servo_worker():
    servo = Servo(
        18,
        min_pulse_width=0.45/1000,
        max_pulse_width=2.45/1000
    )

    print("Servo ready")

    busy_until = 0

    while not stop_event.is_set():
        if not servo_event.wait(timeout=0.2):
            continue

        servo_event.clear()

        now = time.time()
        if now < busy_until:
            continue

        busy_until = now + SERVO_RUN_SECONDS

        start = time.time()

        try:
            while time.time() - start < SERVO_RUN_SECONDS:
                for v in np.linspace(-1, 1, 40):
                    servo.value = v
                    time.sleep(0.03)
                for v in np.linspace(1, -1, 40):
                    servo.value = v
                    time.sleep(0.03)
        finally:
            servo.value = 0  # center

    servo.close()

# ---------------- Detection ----------------
def detection_worker(cam, net, classes):
    process_counter = 0
    window_start = None
    labels_seen = set()
    WINDOW = 1.0

    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            continue

        process_counter += 1

        if PROCESS_EVERY_N and process_counter % PROCESS_EVERY_N != 0:
            continue

        now = time.time()

        if window_start is None:
            window_start = now
            labels_seen.clear()

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        net.setInput(blob)

        try:
            detections = net.forward()
        except:
            continue

        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf < CONF_THRESHOLD:
                continue

            idx = int(detections[0,0,i,1])
            label = classes[idx]

            labels_seen.add(label)
            print("Detected:", label)

        if time.time() - window_start > WINDOW:
            if TARGET_CLASS in labels_seen:
                print("TRIGGER")
                servo_event.set()

            window_start = None
            labels_seen.clear()

# ---------------- Main ----------------
def main():
    CLASSES = ["background","aeroplane","bicycle","bird","boat",
               "bottle","bus","car","cat","chair","cow","diningtable",
               "dog","horse","motorbike","person","pottedplant","sheep",
               "sofa","train","tvmonitor"]

    try:
        net = load_model()

        cam = CameraStream()
        cam.start()

        threading.Thread(target=servo_worker, daemon=True).start()
        threading.Thread(target=detection_worker, args=(cam, net, CLASSES), daemon=True).start()

        print("Running... Ctrl+C to stop")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        traceback.print_exc()
    finally:
        stop_event.set()

if __name__ == "__main__":
    main()