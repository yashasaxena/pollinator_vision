import subprocess
import numpy as np
import cv2

WIDTH, HEIGHT = 640, 480

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

with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p1, \
     subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE) as p2:

    frame_size = WIDTH * HEIGHT * 3

    while True:
        raw = p2.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()