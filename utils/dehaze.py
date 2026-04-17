import cv2
import numpy as np

def get_dark_channel(image, patch_size=15):
    """
    Computes the dark channel for the input image.
    The dark channel is the minimum intensity across all color channels 
    within a local patch around each pixel.
    """
    # Find the minimum pixel intensity across the color channels
    min_channel = np.min(image, axis=2)
    
    # Use a morphological erosion to get the minimum over a local patch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel):
    """
    Estimates the atmospheric light.
    It takes the top 0.1% brightest pixels in the dark channel, and finds 
    the corresponding pixels in the original image. The atmospheric light 
    is calculated by taking the average of these pixels.
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top_pixels = max(int(num_pixels * 0.001), 1)

    # Reshape arrays for sorting
    dark_vec = dark_channel.reshape(-1)
    img_vec = image.reshape(-1, 3)

    # Get the indices of the top 0.1% brightest pixels in the dark channel
    indices = np.argsort(dark_vec)[-num_top_pixels:]

    # Estimate the atmospheric light by taking the mean across these pixels
    atmospheric_light = np.mean(img_vec[indices], axis=0).reshape(1, 1, 3)
    return atmospheric_light

def get_transmission_map(image, atmospheric_light, patch_size=15, omega=0.95):
    """
    Estimates the initial transmission map.
    The transmission describes the portion of light that is not scattered.
    omega introduces a small amount of haze to keep depth perception natural.
    """
    # Normalize the image by the atmospheric light
    norm_image = np.empty(image.shape, dtype=np.float32)
    for c in range(3):
        norm_image[:, :, c] = image[:, :, c] / atmospheric_light[0, 0, c]
    
    # Calculate the dark channel of the normalized image
    norm_dark_channel = get_dark_channel(norm_image, patch_size)
    
    # Transmission calculation
    transmission = 1 - omega * norm_dark_channel
    return transmission

def guided_filter(guide, src, radius=40, epsilon=1e-3):
    """
    Refines the transmission map using a Guided Filter.
    The guide image gives structure/edges to maintain crispness in the src map.
    """
    # Ensure types are float32
    if guide.dtype != np.float32:
        guide = guide.astype(np.float32) / 255.0
    if src.dtype != np.float32:
        src = src.astype(np.float32)
        
    mean_I = cv2.blur(guide, (radius, radius))
    mean_p = cv2.blur(src, (radius, radius))
    mean_Ip = cv2.blur(guide * src, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.blur(guide * guide, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    mean_a = cv2.blur(a, (radius, radius))
    mean_b = cv2.blur(b, (radius, radius))

    # Calculate filtered output
    q = mean_a * guide + mean_b
    return q

def dehaze(image_bgr):
    """
    Main function to apply Dark Channel Prior dehazing on a BGR image.
    Uses all helper functions sequentially to output a dehazed result.
    """
    # Convert image to float32 for calculations (values in [0, 1])
    img_float = image_bgr.astype(np.float32) / 255.0
    
    # 1. Compute Dark Channel
    dark_channel = get_dark_channel(img_float, patch_size=15)
    
    # 2. Estimate Atmospheric Light
    atmospheric_light = get_atmospheric_light(img_float, dark_channel)
    
    # 3. Estimate Transmission Map
    transmission = get_transmission_map(img_float, atmospheric_light, patch_size=15, omega=0.95)
    
    # 4. Refine Transmission Map using Guided Filter
    # Use grayscale image as the guide
    gray_guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined_transmission = guided_filter(gray_guide, transmission, radius=40, epsilon=1e-3)
    
    # Clamp transmission values to avoid dividing by close-to-zero values
    t_min = 0.1
    refined_transmission = cv2.max(refined_transmission, t_min)
    
    # Expand transmission map for 3 channels for broadcasting
    transmission_3d = np.empty_like(img_float)
    for c in range(3):
        transmission_3d[:, :, c] = refined_transmission
        
    # 5. Recover the Dehazed Scene
    # Formula: J(x) = (I(x) - A) / t(x) + A
    J = (img_float - atmospheric_light) / transmission_3d + atmospheric_light
    
    # Clamp values to [0.0, 1.0] and convert back to uint8
    J = np.clip(J, 0.0, 1.0)
    res_bgr = (J * 255).astype(np.uint8)
    
    return res_bgr
