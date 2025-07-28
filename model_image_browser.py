import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(
    page_title="YOLO Image",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS (optional)
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #f8f9fa; padding-top: 2rem; }
    .stFileUploader { margin-top: 1rem; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover { background-color: #45a049; }
    h1, h2, h3, h4, h5, h6 { color: #333333; }
    .stSpinner > div > div { color: #1a73e8; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Model Upload")
model_file = st.sidebar.file_uploader("Upload your YOLO model (*.pt)", type=["pt"])

# st.subheader("Image Upload")
uploaded_images = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

model = None
model_loaded_success = False

if model_file is not None:
    with st.spinner("Loading YOLO model..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_file.write(model_file.getvalue())
            model_path = tmp_file.name

        try:
            model = YOLO(model_path)
            model_loaded_success = True
            st.sidebar.success("YOLO model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
else:
    st.info("👈 Upload a YOLO model (.pt) to get started.")

if model_loaded_success and model is not None:
    if uploaded_images:
        # st.subheader("Results")
        for img_file in uploaded_images:
            img_bytes = img_file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # BGR
            height, width, channel = image.shape

            # st.write("input image size:", height, width, channel)

            with st.spinner(f"Detecting objects in {img_file.name}..."):
                results = model(image, imgsz = (width, height))

            if results:
                result = results[0]
                annotated_image_array = result.plot()  # BGR
                processed_image_rgb = cv2.cvtColor(annotated_image_array, cv2.COLOR_BGR2RGB)

                st.image(processed_image_rgb, use_container_width=True)
            else:
                st.warning(f"No results for {img_file.name}")
    else:
        st.info("Upload image(s) to classify them using the model.")
else:
    if model_file is None:
        st.info("Waiting for a YOLO model to be uploaded...")
