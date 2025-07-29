from ultralytics import YOLO
import cv2


# Load a pretrained classification model or your custom model
model_crane = YOLO("crane.pt")  # Replace with your custom model  

# Initialize webcam
cap = cv2.VideoCapture('D02.mp4')

'''
function to check if the crane is moving
'''
# Create a Background Subtractor using MOG2
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)


def is_moving(frame):
    # Apply the background subtractor to the frame
    fg_mask = background_subtractor.apply(frame)

    # Optional: Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Predict class using YOLO classification
    results = model_crane(frame, imgsz=640)  
    class_id = int(results[0].probs.top1)
    if class_id == 0:   # Assuming class_id 0 corresponds to 'crane moving'
        return True  
    return False


'''
sanity check
'''
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    is_moving(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

