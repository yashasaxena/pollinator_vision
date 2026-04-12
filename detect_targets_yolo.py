#!/usr/bin/env python3
import subprocess
import numpy as np
import os
import signal
from ultralytics import YOLO

# ---------------- Config ----------------
WIDTH, HEIGHT = 640, 480
MODEL_PATH = "best.pt"

CLASSES = ["Bulb", "Ring", "Waffle"]

# ---------------- Kill leftovers ----------------
subprocess.run(["pkill", "-f", "rpicam-vid"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "ffmpeg"], stderr=subprocess.DEVNULL)

# ---------------- Load YOLO ----------------
model = YOLO(MODEL_PATH)

# ---------------- Camera (LOW LATENCY) ----------------
cmd = [
    "rpicam-vid",
    "--timeout", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", "30",
    "--inline",
    "--codec", "h264",
    "--profile", "baseline",
    "--flush",      # 🔥 critical
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

print("Starting camera (detection-only mode)...")

p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, preexec_fn=os.setsid)
p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, preexec_fn=os.setsid)

p1.stdout.close()

frame_size = WIDTH * HEIGHT * 3
frame_count = 0

# ---------------- Main loop ----------------
try:
    while True:
        raw = p2.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

        frame_count += 1

        # 🔥 Only run YOLO every 3 frames (tune this)
        if frame_count % 3 != 0:
            continue

        # 🔥 Resize for speed
        frame_small = frame[::2, ::2]  # faster than cv2.resize (~320x240)

        # Run YOLO
        results = model(frame_small, conf=0.3, imgsz=320, verbose=False)

        # ---------------- Print detections ----------------
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                print(f"{CLASSES[cls]}: {conf:.2f}")

        if len(results) > 0:
            print("----")

except KeyboardInterrupt:
    print("Stopping (Ctrl+C)...")

# ---------------- CLEANUP ----------------
finally:
    print("Cleaning up...")

    for p in (p2, p1):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except:
            pass

    for p in (p2, p1):
        try:
            p.wait(timeout=1)
        except:
            pass

    print("Done.")