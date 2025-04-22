
import streamlit as st
import os
from PIL import Image

# Function to get folders named after YYYYMMDD
def get_date_folders(base_path):
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and len(f) == 8 and f.isdigit()]

# Function to get images from a selected folder
def get_images_from_folder(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]

# Base path where the folders are located
base_path = '/home/byungsoo/Documents/rtsp'

# Streamlit app
st.title('Image Browser')

# Get date folders
date_folders = get_date_folders(base_path)

# Select folder
selected_folder = st.selectbox('Select a date folder', date_folders)

if selected_folder:
    folder_path = os.path.join(base_path, selected_folder)
    images = get_images_from_folder(folder_path)

    # Display images
    for image_path in images:
        image = Image.open(image_path)
        st.image(image, caption=os.path.basename(image_path))

