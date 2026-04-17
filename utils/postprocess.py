import cv2
import numpy as np

def detect_lanes_canny(image_bgr):
    """
    Performs improved lane detection using OpenCV with an updated pipeline for foggy conditions.
    """
    lane_overlay = image_bgr.copy()
    
    # 1. Convert to grayscale 
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian blur (5x5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blurred_clahe = clahe.apply(blurred)
    
    # 4. Adaptive Canny Thresholds based on median
    median_val = np.median(blurred_clahe)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    
    # 5. Apply adaptive Canny edge detection
    edges = cv2.Canny(blurred_clahe, lower, upper)
    
    # 6. Improved ROI Mask (less aggressive, covering the bottom 60%)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    
    # The bottom 60% starts from h * 0.4 downwards
    polygon = np.array([[
        (0, h),
        (0, int(h * 0.4)),
        (w, int(h * 0.4)),
        (w, h)
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 7. Use HoughLinesP
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=50, 
        maxLineGap=150
    )
    
    # 8. Filter lines and draw
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                slope = np.inf
            else:
                slope = (y2 - y1) / (x2 - x1)
                
            if abs(slope) > 0.5:
                cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=4)
                
    # Return the final overlay along with intermediate outputs for debugging
    return lane_overlay, gray, blurred_clahe, edges, masked_edges
