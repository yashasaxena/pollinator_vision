#!/usr/bin/env python3
import os
import subprocess
import signal

def kill_camera_processes():
    # Find any process accessing /dev/video0 (the usual camera)
    try:
        result = subprocess.run(
            ["sudo", "lsof", "/dev/video0"],
            capture_output=True, text=True, check=False
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            print("✅ No active camera processes found.")
            return

        print("🔍 Found camera processes:")
        for line in lines[1:]:
            print(line)
            parts = line.split()
            if len(parts) > 1:
                pid = parts[1]
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"❌ Killed process PID {pid}")
                except Exception as e:
                    print(f"⚠️ Could not kill PID {pid}: {e}")
    except FileNotFoundError:
        print("lsof not installed. Run: sudo apt install lsof")
    except Exception as e:
        print(f"Error while checking camera processes: {e}")

if __name__ == "__main__":
    kill_camera_processes()
