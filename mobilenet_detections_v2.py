#!/usr/bin/env python3
import subprocess
import shutil
import numpy as np
import cv2

WIDTH, HEIGHT, FPS = 640, 480, 30
DURATION_MS = 5000

def find_tool():
    for name in ("libcamera-vid", "raspivid", "rpicam-vid"):
        if shutil.which(name):
            return name
    return None

def build_cmd(tool):
    if tool == "libcamera-vid":
        return [
            tool, "--output", "-", "--timeout", str(DURATION_MS),
            "--width", str(WIDTH), "--height", str(HEIGHT),
            "--framerate", str(FPS)
        ]
    if tool == "raspivid":
        return [
            tool, "-o", "-", "-t", str(DURATION_MS),
            "-w", str(WIDTH), "-h", str(HEIGHT), "-fps", str(FPS)
        ]
    return [tool, "-o", "-", "-t", str(DURATION_MS)]

# Load MobileNet SSD
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

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
ff_cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-f", "h264", "-i", "pipe:0",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
]

print("Running:", " ".join(vid_cmd))

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
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
                    print(label)

    except KeyboardInterrupt:
        pass
    finally:
        ff_proc.terminate()
        vid_proc.terminate()
        ff_proc.wait(timeout=1)
        vid_proc.wait(timeout=1)