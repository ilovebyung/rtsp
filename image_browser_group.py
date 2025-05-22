import streamlit as st
import os
import glob
import re # Import regex
from datetime import datetime
from PIL import Image
import io
from collections import defaultdict # Import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Hourly Image Viewer",
    page_icon="📸",
    layout="wide"
)

# --- Helper Functions ---

def parse_filename_datetime(filename):
    """
    Parses a filename like YYYYMMDD-HHMMSS.jpg to a datetime object.
    Returns None if the filename format doesn't match.
    """
    # Regex to match YYYYMMDD_HHMMSS
    match = re.match(r'(\d{8})_(\d{6})', filename)
    if match:
        try:
            date_str = match.group(1)
            time_str = match.group(2)
            return datetime.strptime(f"{date_str}-{time_str}", '%Y%m%d-%H%M%S')
        except ValueError:
            return None
    return None

def get_date_folders(base_path):
    """
    Get all directories that match the YYYYMMDD pattern.
    """
    date_pattern = os.path.join(base_path, "[0-9]" * 8)
    folders = glob.glob(date_pattern)

    valid_folders = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        if os.path.isdir(folder) and len(folder_name) == 8 and folder_name.isdigit():
            try:
                datetime.strptime(folder_name, '%Y%m%d') # Verify it's a valid date
                valid_folders.append(folder)
            except ValueError:
                continue

    # Sort folders by date (newest first)
    valid_folders.sort(reverse=True)
    return valid_folders

def format_date_for_display(date_str):
    """Formats a date string (YYYYMMDD) for display."""
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return date_obj.strftime('%B %d, %Y') # e.g., "May 21, 2025"

def get_images_and_group_by_hour(folder_path):
    """
    Gets all image files from the specified folder and groups them by hour based on filename.
    Returns a dictionary: {'YYYY-MM-DD HH:00': [{'path': ..., 'datetime': ..., 'filename': ...}, ...]}
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    all_images_data = []

    for extension in image_extensions:
        for img_path in glob.glob(os.path.join(folder_path, extension)):
            filename = os.path.basename(img_path)
            dt_obj = parse_filename_datetime(filename)
            if dt_obj:
                all_images_data.append({'path': img_path, 'datetime': dt_obj, 'filename': filename})
            else:
                st.warning(f"Skipping '{filename}' in '{os.path.basename(folder_path)}' due to invalid naming format (expected YYYYMMDD-HHMMSS).")

    # Sort images by datetime for consistent grouping
    all_images_data.sort(key=lambda x: x['datetime'])

    # Group images by hour
    images_by_hour = defaultdict(list)
    for img_data in all_images_data:
        # Create a key for the hour group: e.g., '2025-05-21 14:00'
        hour_key = img_data['datetime'].strftime('%Y-%m-%d %H:00')
        images_by_hour[hour_key].append(img_data)

    return images_by_hour

# --- Streamlit App ---

# st.title("📸 Hourly Image Browser")
# st.markdown("Select a date folder (e.g., `YYYYMMDD`) and images with `YYYYMMDD-HHMMSS` filenames will be grouped by hour.")

# Sidebar for input and controls
st.sidebar.header("Settings")

# Input for base directory
base_dir = st.sidebar.text_input(
    "Enter the base directory containing date folders:",
    value="./", # Default to current directory
    help="This should be the directory that contains your YYYYMMDD folders"
)

# Check if directory exists
if not os.path.isdir(base_dir):
    st.sidebar.error(f"Directory not found: {base_dir}")
    st.warning(f"The specified directory '{base_dir}' doesn't exist. Please enter a valid directory path.")
else:
    # Get date folders
    date_folders = get_date_folders(base_dir)

    if not date_folders:
        st.warning(f"No date folders (YYYYMMDD format) found in '{base_dir}'.")

        # Show example directory structure
        st.subheader("Expected Directory Structure:")
        st.code("""
        base_directory/
        ├── 20250401/
        │   ├── 20250401-100000.jpg
        │   └── 20250401-111500.jpg
        ├── 20250402/
        │   ├── 20250402-090000.jpg
        │   └── ...
        └── ...
        """)
    else:
        # Create a dictionary of date folders for display
        folder_options = {os.path.basename(folder): folder for folder in date_folders}
        display_options = {format_date_for_display(os.path.basename(folder)): os.path.basename(folder)
                          for folder in date_folders}

        # Date selector
        selected_display_date = st.sidebar.selectbox(
            "Select Date:",
            options=list(display_options.keys()),
            key="date_selector", # Added a key for clarity if multiple selectboxes are used
            help="Choose a date to view images from that day"
        )

        selected_date_folder_name = display_options[selected_display_date]
        selected_date_folder_path = folder_options[selected_date_folder_name]

        # Get images and group them by hour
        images_by_hour = get_images_and_group_by_hour(selected_date_folder_path)

        if not images_by_hour:
            st.info(f"No images with `YYYYMMDD-HHMMSS` naming format found in folder: {selected_date_folder_name}")
        else:
            total_images_count = sum(len(v) for v in images_by_hour.values())
            st.subheader(f"{selected_display_date}: {total_images_count} images found")

            # Image view options (now only grid view for simplicity with grouping)
            st.sidebar.markdown("---")
            st.sidebar.subheader("Display Options")
            cols_per_row = st.sidebar.slider("Images per row:", 1, 4, 3, help="Adjust how many images appear horizontally in the grid.")
            # st.sidebar.info("Images are automatically grouped by hour within the selected date.")

            # Display images in grouped sections
            for hour_key in sorted(images_by_hour.keys()):
                st.markdown(f"---") # Separator for each hour group
                st.header(f"🕔 {hour_key}")
                images_in_hour = images_by_hour[hour_key]

                # Create columns for the images in this hour
                # Ensure we don't try to create more columns than there are images
                current_cols = st.columns(min(cols_per_row, len(images_in_hour)))

                for i, img_data in enumerate(images_in_hour):
                    with current_cols[i % cols_per_row]:
                        try:
                            # For uploaded files, img_data['path'] is a BytesIO object
                            # For local files, it's a string path
                            img_source = img_data['path']
                            if isinstance(img_source, str): # Local file path
                                img = Image.open(img_source)
                            else: # Uploaded file object (BytesIO)
                                img = Image.open(img_source)

                            st.image(img, caption=img_data['filename'], use_container_width=True)

                            # Download button for each image
                            buf = io.BytesIO()
                            img.save(buf, format="PNG") # Always save as PNG for consistent download
                            st.download_button(
                                label="Download",
                                data=buf.getvalue(),
                                file_name=img_data['filename'],
                                mime=f"image/png",
                                key=f"download_{img_data['filename']}_{hour_key}" # Unique key
                            )

                        except Exception as e:
                            st.error(f"Error loading {img_data['filename']}: {str(e)}")

        # Add date navigation (Previous/Next Day)
        st.sidebar.markdown("---")
        # st.sidebar.subheader("Day Navigation")

        # Get current date index
        current_date_idx = date_folders.index(selected_date_folder_path)

        # Previous and next date buttons
        date_cols = st.sidebar.columns(2)

        with date_cols[0]:
            if current_date_idx < len(date_folders) - 1:
                # Navigates to an OLDER date (since folders are sorted newest first)
                next_folder_path = date_folders[current_date_idx + 1]
                next_folder_name = os.path.basename(next_folder_path)
                next_display = format_date_for_display(next_folder_name)
                if st.button(f"⬅️ {next_display}", key="prev_date_btn"):
                    st.session_state.selected_date_name = next_folder_name
                    # st.experimental_rerun() # Use experimental_rerun for immediate update
        with date_cols[1]:
            if current_date_idx > 0:
                # Navigates to a NEWER date
                prev_folder_path = date_folders[current_date_idx - 1]
                prev_folder_name = os.path.basename(prev_folder_path)
                prev_display = format_date_for_display(prev_folder_name)
                if st.button(f"{prev_display} ➡️", key="next_date_btn"):
                    st.session_state.selected_date_name = prev_folder_name
                    # st.experimental_rerun() # Use experimental_rerun for immediate update

        # Initialize session state for selected_date_name if not present
        if 'selected_date_name' not in st.session_state and date_folders:
            st.session_state.selected_date_name = os.path.basename(date_folders[0]) # Default to newest date

# Add footer
# st.sidebar.markdown("---")
# st.sidebar.info(
#     """
#     This app browses images in `YYYYMMDD` folders.
#     Images within each folder are grouped by hour using their `YYYYMMDD-HHMMSS.ext` filename.
#     """
# )

# Hide other elements with custom CSS
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)