import cv2
from ultralytics import YOLO
import asyncio
import util

# Create model
model = YOLO("yolo11s.pt")

# Load RTSP credentials
rtsp_url = util.load_rtsp_credentials("rtsp_config.txt")

# Create capture object
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"Error: Could not open RTSP stream at {rtsp_url}")

async def make_prediction(cap):
    try:
        while True:
            ret, frame = cap.read()
            speeds = []

            if not ret:
                print("End of stream or error")
                break

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
            cv2.imshow('rtsp async', frame)
            
            # Keep this for keyboard interrupts (optional)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("rtsp async inference", sum(speeds)/len(speeds))


asyncio.run(make_prediction(cap))