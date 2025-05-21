import cv2
import numpy as np
import threading
from ultralytics import YOLO
from collections import defaultdict
import util

# Store the track history
track_history = defaultdict(lambda: [])

def create_window():
    window_name = "RTSP Stream"
    cv2.namedWindow(window_name)
    return window_name

def is_window_closed(window_name):
    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        return prop < 1
    except:
        return True

def main():
    rtsp_url = util.load_rtsp_credentials("rtsp_config.txt")

    if not rtsp_url:
        print("Failed to load RTSP credentials. Exiting.")
        return

    # Initialize the new_id variable
    max_id = 0
    new_id = 0

    model = YOLO("yolo11s.pt")  ## Use custon model to detect PPE
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return

        window_name = create_window()

        # Image saving setup
        save_dir = util.save_dir()

        while True:
            if is_window_closed(window_name):
                print("Window closed by user")
                break

            ret, frame = cap.read()

            if ret:
                h,w,c = frame.shape
                # result = model.track(frame, persist=True, imgsz=(720, 1920))[0] ## Update imgsz to yours
                result = model.track(frame, imgsz=(h,w), persist=True)[0] 

                # Get the boxes and track IDs
                if result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    max_id = max(track_ids)
                    track_cls = result.boxes.cls.int().cpu().tolist()


                    # Visualize the result on the frame
                    frame = result.plot()
        
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                        # Save images only when condition is met
                        if (max_id > new_id) and (0 in track_cls)   :  # Assuming 0 is the class ID for the object of interest

                            threading.Thread(target=util.save_tracked, args=(save_dir, frame)).start()
                            new_id = max_id
                            print(f"New ID detected: {new_id}")
                            print(f"Track ID: {track_id}, Class: {model.names[track_cls[0]]}, Coordinates: {box}")
                            print(f"Track history: {track_history[track_id]}")


               # Display the frame
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()