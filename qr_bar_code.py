from pyzbar.pyzbar import decode
from PIL import Image

def extract_barcodes_qr(image_path):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Decode the image
    decoded_objects = decode(image)

    # Prepare results
    results = {'QR_Codes': [], 'Barcodes': []}

    # Iterate through the decoded objects
    for obj in decoded_objects:
        # Check the type of the detected object
        if obj.type == 'QRCODE':
            results['QR_Codes'].append(obj.data.decode('utf-8'))
        else:
            results['Barcodes'].append(obj.data.decode('utf-8'))

    return results

# Example usage for the provided images
image_paths = [
    'temp3.png',  # Replace with the path to your image
]

# Extract barcodes and QR codes from each image
extracted_info = {}
for path in image_paths:
    extracted_info[path] = extract_barcodes_qr(path)
    print(f"Decoded info for {path}: {extracted_info[path]}")  # Debug print statement

print(extracted_info)
