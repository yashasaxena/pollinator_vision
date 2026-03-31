import subprocess

# Capture a still image using the rpicam-still command
def capture_image(filename="test.jpg"):
    try:
        # This calls the same tool as the command line
        subprocess.run(["rpicam-still", "-o", filename, "--immediate"], check=True)
        print(f"Saved image to {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error capturing image: {e}")

capture_image("global_shutter_test.jpg")
