import streamlit as st
import os
import glob
from datetime import datetime
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Date-Based Image Browser",
    page_icon="📅",
    layout="wide"
)

# Add title and description
st.title("📅 Date-Based Image Browser")
st.markdown("Browse images organized in folders by date (YYYYMMDD format)")

# Function to get all date folders
def get_date_folders(base_path):
    # Get all directories that match the YYYYMMDD pattern
    date_pattern = os.path.join(base_path, "[0-9]" * 8)
    folders = glob.glob(date_pattern)
    
    # Filter to ensure they're actually directories and match the date pattern exactly
    valid_folders = []
    for folder in folders:
        folder_name = os.path.basename(folder)
        if os.path.isdir(folder) and len(folder_name) == 8 and folder_name.isdigit():
            try:
                # Verify it's a valid date
                datetime.strptime(folder_name, '%Y%m%d')
                valid_folders.append(folder)
            except ValueError:
                # Skip if it's not a valid date
                continue
    
    # Sort folders by date (newest first)
    valid_folders.sort(reverse=True)
    return valid_folders

# Function to format date for display
def format_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return date_obj.strftime('%B %d, %Y')  # e.g., "January 01, 2023"

# Function to get images from a folder
def get_images_in_folder(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    images = []
    
    for extension in image_extensions:
        images.extend(glob.glob(os.path.join(folder_path, extension)))
        images.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    # Sort images by name
    images.sort()
    return images

# Sidebar for input and controls
st.sidebar.header("Settings")

# Input for base directory
base_dir = st.sidebar.text_input(
    "Enter the base directory containing date folders:",
    value="./",  # Default to current directory
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
        │   ├── image1.jpg
        │   ├── image2.png
        │   └── ...
        ├── 20250402/
        │   ├── image1.jpg
        │   └── ...
        └── ...
        """)
    else:
        # Create a dictionary of date folders for display
        folder_options = {os.path.basename(folder): folder for folder in date_folders}
        display_options = {format_date(os.path.basename(folder)): os.path.basename(folder) 
                          for folder in date_folders}
        
        # Date selector
        selected_display_date = st.sidebar.selectbox(
            "Select Date:",
            options=list(display_options.keys()),
            help="Choose a date to view images from that day"
        )
        
        selected_date_folder = folder_options[display_options[selected_display_date]]
        
        # Get images for the selected date
        images = get_images_in_folder(selected_date_folder)
        
        if not images:
            st.info(f"No images found in folder: {os.path.basename(selected_date_folder)}")
        else:
            # Show number of images found
            st.subheader(f"{selected_display_date}: {len(images)} images found")
            
            # Image view options
            view_mode = st.sidebar.radio(
                "View Mode:",
                options=["Grid View", "Single Image View"]
            )
            
            if view_mode == "Grid View":
                # Grid settings
                cols_per_row = st.sidebar.slider("Images per row:", 1, 6, 3)
                
                # Create grid of images
                cols = st.columns(cols_per_row)
                
                for idx, img_path in enumerate(images):
                    col_idx = idx % cols_per_row
                    
                    with cols[col_idx]:
                        img_filename = os.path.basename(img_path)
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=img_filename, use_container_width=True)
                            
                            # Add a button to view full size
                            if st.button(f"View full size", key=f"view_{idx}"):
                                st.session_state.selected_image = img_path
                                st.session_state.view_mode = "Single Image View"
                                st.rerun()  # Using st.rerun() instead of experimental_rerun
                                
                        except Exception as e:
                            st.error(f"Error loading {img_filename}: {str(e)}")
            
            else:  # Single Image View
                # Initialize selected image if not already set
                if 'selected_image' not in st.session_state or st.session_state.selected_image not in images:
                    st.session_state.selected_image = images[0]
                
                # Get current image index
                current_idx = images.index(st.session_state.selected_image)
                
                # Navigation controls
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    if st.button("⬅️ Previous"):
                        new_idx = (current_idx - 1) % len(images)
                        st.session_state.selected_image = images[new_idx]
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                with col2:
                    st.write(f"Image {current_idx + 1} of {len(images)}")
                
                with col3:
                    if st.button("Next ➡️"):
                        new_idx = (current_idx + 1) % len(images)
                        st.session_state.selected_image = images[new_idx]
                        st.rerun()  # Using st.rerun() instead of experimental_rerun
                
                # Display the selected image
                try:
                    img = Image.open(st.session_state.selected_image)
                    st.image(img, caption=os.path.basename(st.session_state.selected_image), use_container_width=True)
                    
                    # Allow image download
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    
                    st.download_button(
                        label="Download Image",
                        data=buf.getvalue(),
                        file_name=os.path.basename(st.session_state.selected_image),
                        mime=f"image/{img.format.lower() if img.format else 'png'}"
                    )
                    
                    # Show image metadata
                    with st.expander("Image Details"):
                        st.write(f"**Filename:** {os.path.basename(st.session_state.selected_image)}")
                        st.write(f"**Path:** {st.session_state.selected_image}")
                        st.write(f"**Size:** {img.width} x {img.height} pixels")
                        st.write(f"**Format:** {img.format}")
                        if hasattr(img, 'info') and img.info:
                            st.write("**Additional metadata:**")
                            st.write(img.info)
                
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                
                # Return to grid view button
                if st.button("Return to Grid View"):
                    st.session_state.view_mode = "Grid View"
                    st.rerun()  # Using st.rerun() instead of experimental_rerun
        
        # Add date navigation
        st.sidebar.subheader("Date Navigation")
        
        # Get current date index
        current_date_idx = date_folders.index(selected_date_folder)
        
        # Previous and next date buttons
        date_cols = st.sidebar.columns(2)
        
        with date_cols[0]:
            if current_date_idx < len(date_folders) - 1:
                next_folder = os.path.basename(date_folders[current_date_idx + 1])
                next_display = format_date(next_folder)
                if st.button(f"⬅️ {next_display}"):
                    # This navigates to an older date (remember, folders are sorted newest first)
                    st.session_state.selected_date = next_folder
                    st.rerun()  # Using st.rerun() instead of experimental_rerun
        
        with date_cols[1]:
            if current_date_idx > 0:
                prev_folder = os.path.basename(date_folders[current_date_idx - 1])
                prev_display = format_date(prev_folder)
                if st.button(f"{prev_display} ➡️"):
                    # This navigates to a newer date
                    st.session_state.selected_date = prev_folder
                    st.rerun()  # Using st.rerun() instead of experimental_rerun

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    This app browses images in directories organized by date (YYYYMMDD format).
    - Select a date from the dropdown to view images
    - Choose between grid and single image views
    - Navigate between dates using the buttons
    """
)