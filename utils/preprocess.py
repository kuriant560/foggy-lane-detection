import cv2
import numpy as np

# Converts a Streamlit UploadedFile object into a BGR numpy array using OpenCV
def load_image_from_bytes(uploaded_file):
    # Read the file to bytes and convert it into a numpy byte array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the image utilizing OpenCV
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr

# Resizes an image while preserving aspect ratio if its width exceeds max_width
def resize_for_display(image, max_width=800):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / float(w)
        new_h = int(h * ratio)
        # Use INTER_AREA for downsizing 
        image = cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return image

# Converts a BGR numpy array to RGB for displaying properly in Streamlit
def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
