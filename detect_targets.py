#!/usr/bin/env python3
import subprocess
import shutil
import numpy as np
import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import torchvision.transforms.functional as F

# ---------------- Config ----------------
WIDTH, HEIGHT, FPS = 640, 480, 30
MODEL_PATH = "mobilenet_custom_v9.pth"

CLASSES = ["background", "Bulb", "Ring", "Waffle"]
CONF_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    try:
        while True:
            raw = ff_proc.stdout.read(frame_size)
            if not raw:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

            # Optional resize (improves speed)
            frame_small = cv2.resize(frame, (640, 480))

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            tensor = F.to_tensor(frame_rgb).to(device)

            with torch.no_grad():
                output = model([tensor])[0]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Draw detections
            for i in range(len(scores)):
                if scores[i] < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = boxes[i].astype(int)
                label_name = CLASSES[labels[i]]

                # Draw box
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Draw label
                cv2.putText(
                    frame_small,
                    f"{label_name} {scores[i]:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # Show frame
            cv2.imshow("Detection", frame_small)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        ff_proc.terminate()
        vid_proc.terminate()
        ff_proc.wait(timeout=1)
        vid_proc.wait(timeout=1)
        cv2.destroyAllWindows()