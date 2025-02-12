import cv2
from ultralytics import YOLO
import util

def create_window():
    window_name = "RTSP Stream"
    cv2.namedWindow(window_name)
    return window_name

def is_window_closed(window_name):
    try:
        # Try to get window property
        # If window is closed, this will raise an error
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        return prop < 1
    except:
        return True

def main():
    # Load RTSP credentials
    rtsp_url = util.load_rtsp_credentials("rtsp_config.txt")

    if not rtsp_url:
        print("Failed to load RTSP credentials. Exiting.")
        return

    # Create model
    model = YOLO("yolo11s.pt")
    
    try:
        # Create capture object
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return

        # Create named window
        window_name = create_window()

        while True:
            # Check if window was closed
            if is_window_closed(window_name):
                print("Window closed by user")
                break

            ret, frame = cap.read()

            if not ret:
                print("End of stream or error")
                break

            # Run YOLO detection
            results = model(frame)

            # Process detections
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if cls == 0:  # Assuming class 0 is what you want to detect
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f'{model.names[cls]} {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Show frame
            cv2.imshow(window_name, frame)
            
            # Keep this for keyboard interrupts (optional)
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