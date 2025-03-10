from PIL import Image

def make_reddish(image_path, output_path):
    """
    Turns an image into a reddish color image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """
    try:
        img = Image.open(image_path).convert("RGB")  # Ensure RGB mode

        pixels = img.load()
        width, height = img.size

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                # Increase red, decrease green and blue
                new_r = min(255, int(r * 1.5))  # Increase red, but cap at 255
                new_g = max(0, int(g * 0.7))    # Decrease green, but cap at 0
                new_b = max(0, int(b * 0.7))    # Decrease blue, but cap at 0

                pixels[x, y] = (new_r, new_g, new_b)

        img.save(output_path)
        print(f"Reddish image saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# make_reddish("input.jpg", "output_reddish.jpg")

def make_reddish_alternative(image_path, output_path, red_factor = 1.3, green_factor = 0.8, blue_factor = 0.8):
    """
    Turns an image into a reddish color image, with adjustable factors.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        red_factor (float): factor to multiply red channel by.
        green_factor (float): factor to multiply green channel by.
        blue_factor (float): factor to multiply blue channel by.
    """
    try:
        img = Image.open(image_path).convert("RGB")

        pixels = img.load()
        width, height = img.size

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]

                new_r = min(255, int(r * red_factor))
                new_g = max(0, int(g * green_factor))
                new_b = max(0, int(b * blue_factor))

                pixels[x, y] = (new_r, new_g, new_b)

        img.save(output_path)
        print(f"Reddish image saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example of alternative usage:
# make_reddish_alternative("input.jpg", "output_reddish.jpg", 1.8, 0.6, 0.6)

input = '/home/byungsoo/Documents/rtsp/20250310/detected_20250310_222341.jpg'
make_reddish(input, "output_reddish.jpg")
make_reddish_alternative(input, "output_reddish.jpg", 1.8, 0.6, 0.6)


import cv2
import numpy as np
import os

def make_reddish_opencv(image_path, output_path, red_factor=1.5, green_factor=0.7, blue_factor=0.7):
    """
    Turns an image into a reddish color image using OpenCV.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        red_factor (float): Factor to multiply the red channel by.
        green_factor (float): Factor to multiply the green channel by.
        blue_factor (float): Factor to multiply the blue channel by.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Convert to float for accurate calculations
        img_float = img.astype(np.float32)

        # Apply the color manipulation
        img_float[:, :, 2] *= red_factor  # Red channel (OpenCV uses BGR)
        img_float[:, :, 1] *= green_factor  # Green channel
        img_float[:, :, 0] *= blue_factor  # Blue channel

        # Clip values to the valid range [0, 255]
        img_float = np.clip(img_float, 0, 255)

        # Convert back to uint8
        img_reddish = img_float.astype(np.uint8)

        # Save the output image
        cv2.imwrite(output_path, img_reddish)
        print(f"Reddish image saved to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
make_reddish_opencv(input, "output_reddish_cv2.jpg")
make_reddish_opencv(input, "output_reddish_cv2_alt.jpg", 1.8, 0.6, 0.6)