import cv2
import numpy as np
import threading
from ultralytics import YOLO
import util

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

    model = YOLO("yolo11s.pt")
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        speeds = []
        
        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return

        window_name = create_window()

        # Image saving setup
        save_dir = util.save_dir()
        last_save_time = 0

        while True:
            if is_window_closed(window_name):
                print("Window closed by user")
                break

            ret, frame = cap.read()

            if ret:

                results = model(frame)
                speeds.append(results[0].speed['inference'])

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        conf = box.conf[0]
                        cls = int(box.cls[0]) # person class
                        if cls == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f'{model.names[cls]} {conf:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                            # util.save_detected(save_dir, frame, last_save_time)
                            threading.Thread(target=util.save_detected, args=(save_dir, frame, last_save_time)).start()

                            # Convert to float for accurate calculations
                            frame = frame.astype(np.float32)

                            # Apply the color manipulation
                            frame[:, :, 2] *= 1.4  # Red channel (OpenCV uses BGR)
                            frame[:, :, 1] *= 0.8  # Green channel
                            frame[:, :, 0] *= 0.8  # Blue channel

                            # Clip values to the valid range [0, 255]
                            frame = np.clip(frame, 0, 255)

                            # Convert back to uint8
                            frame = frame.astype(np.uint8)

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
        if len(speeds) > 0:
            print("PPE inference", sum(speeds)/len(speeds))
        else:
            print("No inference data available")

if __name__ == "__main__":
    main()