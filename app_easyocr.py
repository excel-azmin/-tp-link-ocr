import easyocr
import cv2
from PIL import Image
import numpy as np

# Initialize EasyOCR reader with CPU (gpu=False)
reader = easyocr.Reader(['en'], gpu=False)

# Image Preprocessing Function
def preprocess_image(image_path, output_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise the image using Gaussian Blur
    denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Apply adaptive thresholding to enhance text visibility
    thresh_img = cv2.adaptiveThreshold(denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    # Resize (Zoom) the image for better OCR accuracy (zoom factor 2x)
    resized_img = cv2.resize(thresh_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Save the preprocessed image (for visualization purposes)
    cv2.imwrite(output_path, resized_img)

    return output_path

# Perform OCR using EasyOCR after preprocessing
def perform_ocr(preprocessed_image_path):
    # Perform OCR on the preprocessed image
    result = reader.readtext(preprocessed_image_path, detail=1)

    # Extract text from the result
    extracted_text = [item[1] for item in result]

    # Print the extracted text line by line
    print("Extracted Text (Line by Line):")
    for i, text in enumerate(extracted_text, start=1):
        print(f"Line {i}: {text}")

    # Optionally, filter text by confidence score (e.g., confidence > 0.5)
    print("\nFiltered Text (Confidence > 0.5):")
    filtered_text = [item[1] for item in result if item[2] > 0.5]
    for i, text in enumerate(filtered_text, start=1):
        print(f"Line {i}: {text}")

    return extracted_text, filtered_text

# Main function
if __name__ == "__main__":
    # Input image path (the low-quality image)
    image_path = 'temp3.png'

    # Output preprocessed image path
    preprocessed_image_path = 'preprocessed_image.jpg'

    # Step 1: Preprocess the image
    preprocess_image(image_path, preprocessed_image_path)

    # Step 2: Perform OCR on the preprocessed image
    perform_ocr(preprocessed_image_path)
