import cv2
from ultralytics import YOLO
import threading

def check_status():
    ret, image = cap.read()
    return image


def check_speed():
    """Monitors check_status() and skips it if it exceeds the timeout."""
    timeout = 0.1
    image = None
    event = threading.Event()

    def run_with_timeout():
        nonlocal image
        try:
            image = check_status()
        finally:
            event.set()

    thread = threading.Thread(target=run_with_timeout)
    thread.start()
    event.wait(timeout)

    if event.is_set():
        if image is not None:
            return image
        else:
            print("Status check completed within timeout, but no result returned") # in case check_status returns None.
    else:
        print("Status check timed out.")
        return None

# Create model
model = YOLO("yolo11n.pt")

try:
    # Create capture object
    cap = cv2.VideoCapture(0)
    speeds = []

    while True:
        frame = check_speed()
        if frame is not None:

            # Run YOLO detection
            results = model(frame)
            speeds.append(results[0].speed['inference'])

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
            cv2.imshow('timeout_inference', frame)
            
            # Keep this for keyboard interrupts (optional)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("timeout inference", sum(speeds)/len(speeds))


