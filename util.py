import time
import os
import cv2

# Configuration file path
CONFIG_FILE = "rtsp_config.txt"

''' 
192.168.100.100:554/stream_ch00_0
id
password
'''

# ffmpeg -f v4l2 -video_size 640x480 -i /dev/video0 -c:v libx264 -preset medium -crf 23 -f rtp rtp://127.0.0.1:5000

def load_rtsp_credentials(filepath=CONFIG_FILE):
    """Loads RTSP URL, ID, and password from a text file."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            if len(lines) != 3:
                raise ValueError("Config file must contain 3 lines: URL, ID, Password")

            url = lines[0].strip()
            id = lines[1].strip()
            password = lines[2].strip()

            # Construct the RTSP URL
            rtsp_url = f"rtsp://{id}:{password}@{url}"
            return rtsp_url

    except FileNotFoundError:
        print(f"Error: Config file '{filepath}' not found.")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def save_dir():
    save_dir = time.strftime("%Y%m%d")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

# def save_detected(save_dir, frame, last_save_time):
#         # Save image every second when a person is detected
#     current_time = time.time()
#     if current_time - last_save_time >= 1:
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         image_path = os.path.join(save_dir, f"detected_{timestamp}.jpg")
#         cv2.imwrite(image_path, frame)
#         last_save_time = current_time


frame_count = 1  # Initialize frame counter

def save_detected(save_dir, frame, frame_count):
    # Save every 20th image when a person is detected
    if frame_count % 20 == 0:  # Check if the frame count is a multiple of 20
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(save_dir, f"detected_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

    return frame_count + 1  # Increment frame count


