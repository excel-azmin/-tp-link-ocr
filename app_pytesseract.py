import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import os

# Set the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # or the actual path to tesseract on your system

def preprocess_image(image_path, zoom_factor=2):
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found. Please check the file path.")

    # Load the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Check if the image was successfully loaded
    if img is None:
        raise ValueError(f"Could not load image. Please check the file integrity or format of '{image_path}'.")

    # Zoom (Resize) the image
    height, width = img.shape[:2]
    img_zoomed = cv2.resize(img, (width * zoom_factor, height * zoom_factor), interpolation=cv2.INTER_LINEAR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Increase contrast by histogram equalization
    contrast = cv2.equalizeHist(denoised)

    # Sharpen the image using a kernel filter
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)

    # Apply adaptive thresholding to make the image binary (black-and-white)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Optionally save the preprocessed image to visualize the result
    cv2.imwrite('preprocessed_image_zoomed.png', thresh)

    return thresh

def ocr_image(image_path, zoom_factor=2):
    # Preprocess the image for better OCR results (with zoom)
    preprocessed_image = preprocess_image(image_path, zoom_factor=zoom_factor)

    # Set Tesseract configuration (Optional: Use PSM and whitelist for better accuracy)
    custom_config = r'--oem 3 --psm 6'  # OEM 3 = Default + LSTM, PSM 6 = Assume a single uniform block of text

    # Use pytesseract to extract text
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

    return text

# Load the low-quality image and perform OCR
image_path = 'photo_2024-10-21_12-06-23 (2).jpg'

try:
    extracted_text = ocr_image(image_path, zoom_factor=2)  # Set zoom_factor (e.g., 2 for 2x zoom)
    print("Extracted Text:")
    print(extracted_text)
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
