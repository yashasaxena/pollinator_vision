#!/usr/bin/env python3
"""
Forcefully reset Raspberry Pi camera and GPIO state.
Run this when your main script crashes or camera won't reopen.
"""

import os
import signal
import subprocess
import time
import RPi.GPIO as GPIO

# GPIO pin used by LED (update if different)
LED_PIN = 17

def kill_camera_processes():
    """Kill any processes holding /dev/video0."""
    print("🔍 Checking for processes using /dev/video0 ...")
    try:
        result = subprocess.run(
            ["sudo", "lsof", "/dev/video0"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            print("✅ No active camera processes found.")
            return

        for line in lines[1:]:
            parts = line.split()
            if len(parts) > 1:
                pid = int(parts[1])
                print(f"❌ Killing process PID {pid}: {parts[0]}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print(f"⚠️ Could not kill PID {pid}: {e}")
    except Exception as e:
        print(f"Error checking camera processes: {e}")

def reset_gpio():
    """Turn off LED and reset GPIO state."""
    print("🧹 Cleaning up GPIO...")
    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("✅ GPIO cleaned up successfully.")
    except Exception as e:
        print(f"⚠️ GPIO cleanup error: {e}")

def reload_camera_driver():
    """Reload the bcm2835-v4l2 driver (for CSI camera)."""
    print("🔁 Reloading bcm2835-v4l2 driver (if loaded)...")
    try:
        subprocess.run(["sudo", "modprobe", "-r", "bcm2835_v4l2"], check=False)
        time.sleep(0.5)
        subprocess.run(["sudo", "modprobe", "bcm2835_v4l2"], check=False)
        print("✅ Driver reload attempted.")
    except Exception as e:
        print(f"⚠️ Driver reload failed: {e}")

def main():
    kill_camera_processes()
    reset_gpio()
    reload_camera_driver()
    print("✅ Reset complete. Try rerunning your detection script now.")

if __name__ == "__main__":
    main()
