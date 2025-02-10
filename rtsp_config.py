import cv2
from ultralytics import YOLO
import numpy as np
import util



# Load RTSP credentials
rtsp_url = util.load_rtsp_credentials("rtsp_config.txt")

if rtsp_url:
    # model = YOLO("yolo11s.pt")
    model = YOLO("yolov8s.pt")  # Or your preferred model path

    try:
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            exit()  # Exit the program if stream cannot be opened

        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of stream or error")
                break

            results = model(frame)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    if cls == 0:  # Assuming class 0 is what you want to detect
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f'{model.names[cls]} {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("RTSP Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'cap' in locals() and cap is not None:  # Check if cap is defined
            cap.release()
        cv2.destroyAllWindows()

else:
    print("Failed to load RTSP credentials. Exiting.")