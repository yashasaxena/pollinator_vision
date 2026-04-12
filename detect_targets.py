#!/usr/bin/env python3
import subprocess
import shutil
import numpy as np
import cv2
import torch
import time
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import torchvision.transforms.functional as F

# ---------------- Config ----------------
WIDTH, HEIGHT, FPS = 640, 480, 30
MODEL_PATH = "mobilenet_custom_v9.pth"

CLASSES = ["background", "Bulb", "Ring", "Waffle"]
CONF_THRESHOLD = 0.5

FRAME_SKIP = 5  # process every N frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Performance tweaks ----------------
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# ---------------- Camera ----------------
def find_tool():
    for name in ("libcamera-vid", "rpicam-vid", "raspivid"):
        if shutil.which(name):
            return name
    return None

def build_cmd(tool):
    return [
        tool,
        "--output", "-",
        "--timeout", "0",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "h264",
        "--nopreview"
    ]

# ---------------- Model ----------------
def load_model():
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        num_classes=4
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------- Camera setup ----------------
tool = find_tool()
if not tool:
    print("No camera tool found")
    exit(1)

vid_cmd = build_cmd(tool)
ff_cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "h264", "-i", "pipe:0",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"
]

print("Starting camera...")

with subprocess.Popen(vid_cmd, stdout=subprocess.PIPE) as vid_proc, \
     subprocess.Popen(ff_cmd, stdin=vid_proc.stdout, stdout=subprocess.PIPE) as ff_proc:

    vid_proc.stdout.close()
    frame_size = WIDTH * HEIGHT * 3

    frame_count = 0

    try:
        while True:
            raw = ff_proc.stdout.read(frame_size)
            if not raw:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # ---------------- Resize for speed ----------------
            frame_small = cv2.resize(frame, (320, 320))

            # ---------------- Preprocess ----------------
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(frame_rgb).to(device)

            # ---------------- Inference ----------------
            try:
                output = model([tensor])[0]
            except Exception as e:
                print("Inference error:", e)
                continue

            boxes = output.get("boxes", []).cpu().numpy() if "boxes" in output else []
            scores = output.get("scores", []).cpu().numpy() if "scores" in output else []
            labels = output.get("labels", []).cpu().numpy() if "labels" in output else []

            # ---------------- Detection logic ----------------
            detected = False

            for i in range(min(5, len(scores))):
                if scores[i] < CONF_THRESHOLD:
                    continue

                detected = True
                label_name = CLASSES[labels[i]]
                print(f"{label_name}: {scores[i]:.2f}")

            if detected:
                print("----")

            # ---------------- Prevent CPU overload ----------------
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        ff_proc.terminate()
        vid_proc.terminate()
        ff_proc.wait(timeout=1)
        vid_proc.wait(timeout=1)