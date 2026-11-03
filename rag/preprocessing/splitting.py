# use OpenCV contour detection and adaptive thresholding to find bounding boxes around non-text elements.

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

def extract_visual_crops(page_image_path):
    """
    Detects diagrams, sketches, and boxed sections using OpenCV contours
    and returns image crops alongside their coordinates.
    """
    image = cv2.imread(page_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarize and invert to isolate ink/drawings from white background
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Use morphological operations to connect nearby strokes into blobs/regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    visual_crops = []
    h_orig, w_orig = image.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out tiny noise and full-page bounding boxes
        if w > 100 and h > 100 and (w * h) < (w_orig * h_orig * 0.8):
            crop = image[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            visual_crops.append({
                "box": (x, y, w, h),
                "image": Image.fromarray(crop_rgb)
            })
            
    return visual_crops