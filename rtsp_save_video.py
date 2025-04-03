import cv2
import os
import time # Optional: for adding timestamp to filename

# --- Configuration ---
RTSP_USERNAME = "id"
RTSP_PASSWORD = "pw"

# --- !! IMPORTANT: Replace these placeholders !! ---
CAMERA_IP_ADDRESS = "YOUR_CAMERA_IP_ADDRESS"  # e.g., "192.168.1.100"
RTSP_PORT = "554"                          # Default RTSP port is 554, change if needed
STREAM_PATH = "YOUR_STREAM_PATH"           # e.g., "/stream1", "/live", "/video.sdp", "/cam/realmonitor?channel=1&subtype=0"
# Check your camera's documentation for the correct stream path!
# --- End of Placeholders ---

OUTPUT_FILENAME = "saved_rtsp_stream.mp4"
# Optional: Add timestamp to filename to avoid overwriting
# timestamp = time.strftime("%Y%m%d_%H%M%S")
# OUTPUT_FILENAME = f"saved_rtsp_stream_{timestamp}.mp4"

DEFAULT_FPS = 10  # FPS to use if it cannot be determined from the stream

# Construct the RTSP URL with credentials
rtsp_url = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{CAMERA_IP_ADDRESS}:{RTSP_PORT}/{STREAM_PATH}"

print(f"Attempting to connect to: rtsp://{RTSP_USERNAME}:****@{CAMERA_IP_ADDRESS}:{RTSP_PORT}/{STREAM_PATH}")

cap = None
out = None

try:
    # 1. Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url) # Can also try adding cv2.CAP_FFMPEG as a second argument if default fails

    if not cap.isOpened():
        raise IOError(f"Error: Could not open RTSP stream. Check URL, credentials, network, and camera settings.\nURL Attempted: {rtsp_url}")
    else:
        print("Successfully connected to RTSP stream.")

    # 2. Get video properties from the stream
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_width <= 0 or frame_height <= 0:
         print("Warning: Could not get valid frame dimensions from stream. Trying to read a frame first...")
         ret, frame = cap.read()
         if ret:
             frame_height, frame_width = frame.shape[:2]
             print(f"Got dimensions from first frame: {frame_width}x{frame_height}")
         else:
             raise IOError("Could not read initial frame to determine dimensions.")


    if source_fps <= 0:
        print(f"Warning: Could not get valid FPS from source ({source_fps}). Using default: {DEFAULT_FPS}")
        fps_to_use = DEFAULT_FPS
    else:
        fps_to_use = source_fps
        print(f"Source FPS detected: {fps_to_use}")

    # 3. Define the codec and create VideoWriter object
    # 'mp4v' is common for .mp4 files. Others: 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(f"Initializing VideoWriter: File='{OUTPUT_FILENAME}', Codec='mp4v', FPS={fps_to_use}, Size=({frame_width}x{frame_height})")
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps_to_use, (frame_width, frame_height))

    if not out.isOpened():
        raise IOError(f"Error: Could not open VideoWriter for file '{OUTPUT_FILENAME}'. Check permissions and codec support.")
    else:
        print(f"Recording started. Saving video to '{OUTPUT_FILENAME}'. Press 'q' in the display window to stop.")


    # 4. Read frames and write to file
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream ended or connection lost.")
            break

        # Write the frame to the output file
        out.write(frame)

        # Optional: Display the stream (can be commented out if running headless)
        cv2.imshow('RTSP Stream (Saving)', frame)

        # Check for 'q' key press to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Stopping recording.")
            break

except IOError as e:
    print(e) # Print specific IO errors
except KeyboardInterrupt:
    print("\nRecording interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # 5. Release resources
    print("Cleaning up...")
    if cap is not None and cap.isOpened():
        cap.release()
        print("VideoCapture released.")
    if out is not None and out.isOpened():
        out.release()
        print(f"VideoWriter released. Video saved as '{OUTPUT_FILENAME}' (if recording started).")
    elif out is not None:
        print("VideoWriter was initialized but may not have been opened or released correctly.")

    cv2.destroyAllWindows()
    print("OpenCV windows closed.")
    print("Script finished.")