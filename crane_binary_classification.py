##############################
# YOLO Binary Classification #
##############################

"""
# 0. Import the YOLO class from the ultralytics library
"""

from ultralytics import YOLO


"""
# 1. Data Preparation (Crucial Step - Do this before running the script) ---
Your dataset should be organized as follows for crane classification:
"""

# dataset/
# ├── train/
# │   ├── moving/
# │   │   ├── img1.jpg
# │   │   ├── img2.jpg
# │   │   └── ...
# │   └── not_moving/
# │       ├── imgA.jpg
# │       ├── imgB.jpg
# │       └── ...
# ├── val/
# │   ├── moving/
# │   │   ├── imgX.jpg
# │   │   └── ...
# │   └── not_moving/
# │       ├── imgY.jpg
# │       └── ...
# └── test/ (Optional, for final evaluation)
#     ├── moving/
#     │   ├── imgZ.jpg
#     │   └── ...
#     └── not_moving/
#         ├── imgW.jpg
#         └── ...
#

# Define the path to your dataset root directory.
# This directory should contain 'train', 'val', and 'test' (optional) subdirectories,
# each containing subdirectories for your classes ('moving', 'not_moving').
dataset_path = 'dataset/' # IMPORTANT: Update this to your actual dataset path

# Define the path to your samples folder for prediction sanity check.
# This folder should contain images you want to test the trained model on.
samples_folder = 'samples/' # IMPORTANT: Update this to your actual samples path

"""## 2. Load a pre-trained YOLO model for classification"""

# For classification, you typically start with a pre-trained classification model.
# 'yolov8n-cls.pt' is the nano version, good for quick testing.
# You can choose larger models like 'yolov8s-cls.pt', 'yolov8m-cls.pt', etc.,
# depending on your computational resources and desired accuracy.
print("\n Loading crane classification model...")
model = YOLO('yolo11s-cls.pt')

"""## 3. Train the model  """

# The 'data' argument points to the root directory of your dataset.
# The 'epochs' argument specifies the number of training epochs.
# 'imgsz' sets the input image size.
# 'patience' stops training if validation metric does not improve for 'patience' epochs.
# 'batch' sets the batch size. Adjust based on your GPU memory.
# 'name' sets the name of the experiment directory inside 'runs/classify/'.
print(f"\nStarting model training on dataset: {dataset_path}")
results = model.train(
    data=dataset_path,
    epochs=40, # You might need more epochs for a real dataset
    imgsz=128, # Adjust image size as needed, common sizes are 64, 128, 224
    patience=3, # Stop if no improvement for 3 epochs
    batch=16, # Adjust based on your hardware
    name='yolo_binary_classification_run'
)
print("\n Model training complete.")
print(f"Training results saved to: {model.trainer.save_dir}")

"""## 4. Make predictions on sample images  """

# The 'source' argument can be a single image, a directory of images, or a video.
# The 'save' argument will save the prediction results (e.g., images with predictions)
# to a 'runs/classify/predict' directory.
print(f"\n Making predictions on sample images in: {samples_folder}")
predictions = model.predict(
    source=samples_folder,
    save=True, # Save predicted images with labels
    name='yolo_binary_classification_predict'
)
print("\n Prediction complete.")
print(f"Prediction results saved to: {model.predictor.save_dir}")

# --- Optional: Print prediction details ---
print("\n--- Prediction Details ---")
for i, pred in enumerate(predictions):
    # 'pred' is a Results object for each image
    # For classification, 'probs' contains the probabilities for each class
    # 'names' maps class indices to class names
    if hasattr(pred, 'probs') and pred.probs is not None:
        top_prob_index = pred.probs.top1
        top_prob_value = pred.probs.top1conf.item()
        predicted_class_name = model.names[top_prob_index]
        print(f"Image {i+1}: Predicted Class: {predicted_class_name}, Confidence: {top_prob_value:.4f}")
    else:
        print(f"Image {i+1}: Could not retrieve prediction probabilities.")

print("\nBinary classification process completed.")

# --- Cleanup (Optional) ---
# Remove dummy directories and files after demonstration
# if os.path.exists(dataset_path):
#     shutil.rmtree(dataset_path)
#     print(f"Removed dummy dataset structure: {dataset_path}")
# if os.path.exists(samples_folder):
#     shutil.rmtree(samples_folder)
#     print(f"Removed dummy samples folder: {samples_folder}")