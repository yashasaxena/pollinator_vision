#!/usr/bin/env python3
import subprocess
import numpy as np
import cv2
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

# ---------------- Camera (LOW LATENCY SETTINGS) ----------------
cmd = [
    "rpicam-vid",
    "--timeout", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", "30",
    "--inline",
    "--codec", "h264",
    "--profile", "baseline",
    "--flush",          # 🔥 critical for low latency
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

print("Starting camera (low latency mode)...")

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

        # 🔥 Always show frame (smooth video)
        display_frame = cv2.resize(frame, (320, 320))

        # 🔥 Only run YOLO every 3 frames
        if frame_count % 3 == 0:
            results = model(display_frame, conf=0.3, imgsz=320, verbose=False)

            # Draw detections
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = CLASSES[cls]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("YOLO Low-Latency Feed", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    cv2.destroyAllWindows()
    print("Done.")