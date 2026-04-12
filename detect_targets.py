#!/usr/bin/env python3
import subprocess
import numpy as np
import cv2
import os
import signal
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import torchvision.transforms.functional as F

device = torch.device("cpu")  # Pi = CPU

model = fasterrcnn_mobilenet_v3_large_fpn(
    weights=None,
    num_classes=4  # background + 3 classes
)

model.load_state_dict(torch.load("mobilenet_custom_v9.pth", map_location=device))
model.to(device)
model.eval()

WIDTH, HEIGHT = 640, 480

CLASSES = ["background", "Bulb", "Ring", "Waffle"]
CONF_THRESHOLD = 0.5

# ---------------- Kill leftovers (important) ----------------
subprocess.run(["pkill", "-f", "rpicam-vid"], stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "ffmpeg"], stderr=subprocess.DEVNULL)

# ---------------- Camera pipeline ----------------
cmd = [
    "rpicam-vid",
    "--timeout", "0",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--inline",
    "--codec", "h264",
    "--nopreview",
    "-o", "-"
]

ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "quiet",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-f", "h264",
    "-i", "pipe:0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

print("Starting camera...")

# 🔥 Start processes in their own groups
p1 = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    preexec_fn=os.setsid
)

p2 = subprocess.Popen(
    ffmpeg_cmd,
    stdin=p1.stdout,
    stdout=subprocess.PIPE,
    preexec_fn=os.setsid
)

p1.stdout.close()

frame_size = WIDTH * HEIGHT * 3

# ---------------- Main loop ----------------
frame_count = 0

try:
    while True:
        raw = p2.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

        frame_count += 1

        # ---------------- Run model every 10th frame ----------------
        if frame_count % 10 == 0:
            frame_small = cv2.resize(frame, (320, 320))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            tensor = F.to_tensor(frame_rgb).to(device)

            with torch.no_grad():
                output = model([tensor])[0]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for i in range(min(5, len(scores))):
                if scores[i] < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                label_name = CLASSES[labels[i]]

                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame_small,
                    f"{label_name} {scores[i]:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Camera", frame_small)

        else:
            cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping (Ctrl+C)...")

# ---------------- CLEANUP (CRITICAL) ----------------
finally:
    print("Cleaning up camera processes...")

    for p in (p2, p1):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass

    for p in (p2, p1):
        try:
            p.wait(timeout=1)
        except Exception:
            pass

    cv2.destroyAllWindows()

    print("Done.")