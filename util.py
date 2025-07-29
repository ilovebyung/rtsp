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

frame_count = 0  # Initialize frame counter

def save_detected(save_dir, frame, frame_count):
    # Save every 20th image when a person is detected
    if frame_count % 20 == 0:  # Check if the frame count is a multiple of 20
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(save_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

    return frame_count + 1  # Increment frame count


def save_tracked(save_dir, frame):
    # Save new detected  
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(save_dir, f"{timestamp}.jpg")
    cv2.imwrite(image_path, frame)


# Initialize webcam
cap = cv2.VideoCapture(0)

# Create MOG2 background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply background subtraction 
    fgmask = fgbg.apply(gray)
    # Apply morphological operations to reduce noise
    fgmask_cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # Bitwise AND to isolate moving pixels in grayscale
    moving_pixels_gray = cv2.bitwise_and(gray, fgmask_cleaned)

    # Display result
    cv2.imshow("Grayscale Moving Pixels (MOG2)", moving_pixels_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Load RTSP credentials
        rtsp_url = load_rtsp_credentials()
        if not rtsp_url:
            raise ValueError("Failed to load RTSP credentials.")

        # Open video capture
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise ValueError("Failed to open video capture.")

        window_name = "RTSP Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Initialize model and track history
        model = ...  # Load your tracking model here
        track_history = {}
        new_id = -1  # Initialize new ID

        save_dir = util.save_dir()  # Get save directory