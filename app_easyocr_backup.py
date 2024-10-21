import easyocr
# Initialize reader with CPU only (gpu=False)
reader = easyocr.Reader(['en'], gpu=False)

# Perform OCR on the image
result = reader.readtext('temp2.png', detail=1)

# Extract text from the result
extracted_text = [item[1] for item in result]

# Print the extracted text line by line
print("Extracted Text (Line by Line):")
for i, text in enumerate(extracted_text, start=1):
    print(f"Line {i}: {text}")
