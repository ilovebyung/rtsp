from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Load a pretrained classification model or your custom model
model = YOLO("crane.pt")  # Replace with your custom model  

# Path to the folder containing your sample images
image_folder = 'samples'
files = os.listdir(image_folder)

for file in files:
    image_path = os.path.join(image_folder, file)
    image = cv2.imread(image_path)

    # Predict class using YOLO classification
    results = model(image)

    # Get class name and confidence
    class_id = int(results[0].probs.top1)
    class_name = results[0].names[class_id]
    confidence = results[0].probs.top1conf

    # Display image and prediction
    # plt.imshow(image)
    print(f"Class: {class_name} | Confidence: {confidence * 100, 2}%")


# Example usage with a specific image file
file = '/home/byungsoo/Documents/rtsp/samples/frame_0019.jpg'
image = cv2.imread(file)
results = model(image, imgsz=640) 
plt.imshow(image)
# Get class name and confidence
class_id = int(results[0].probs.top1)
class_name = results[0].names[class_id]
confidence = results[0].probs.top1conf
print(f"Class: {class_name} | Confidence: {confidence * 100, 2}%")


def is_moving(image_path):
    image = cv2.imread(image_path)
    results = model(image, imgsz=640)  # Adjust image size if necessary
    class_id = int(results[0].probs.top1)
    if class_id == 0:   # Assuming class_id 0 corresponds to 'crane'
        return True  
    return False
    
file = 'samples/frame_0019.jpg' # moving crane
file = 'samples/frame_0678.jpg' # not moving crane
is_moving(file)