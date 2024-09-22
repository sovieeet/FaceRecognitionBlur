import cv2
import os
from utils.MTCNN_settings import MTCNN_setup
from utils.face_processing import process_face_image
from utils.gpu_check import setup_device
import sys
import time
"""
Face blur using image
"""
# Check if TensorFlow detect GPU and setup the memory growth
setup_device()

# Call MTCNN function to create the detector
detector = MTCNN_setup()

# Image path
image_path = 'face_blur/images/kanye-taylor.jpg' # You can change the path to any image you want to process

image = cv2.imread(image_path)

if image is None:
    print(f"Image can't be loaded: {image_path}")
    sys.exit()

output = process_face_image(image, detector)

# Show proceseed image
cv2.imshow('Processed Image', output)
cv2.waitKey(0)  # Wait until any key is pressed
cv2.destroyAllWindows()

# Create the results directory if it doesn't exist
output_dir = 'results/images'  # Save processed images in results/images
os.makedirs(output_dir, exist_ok=True)

# Save the processed image with a unique name
timestamp = int(time.time())
filename = f'image_{timestamp}.jpg'
output_path = os.path.join(output_dir, filename)
cv2.imwrite(output_path, output)
print(f"Image processed and saved as '{output_path}'")
