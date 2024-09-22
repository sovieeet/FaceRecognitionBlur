import cv2
import os
from utils.MTCNN_settings import MTCNN_setup
from utils.face_processing import process_face_image
from utils.gpu_check import setup_device
import time
"""
Face blur using webcam
"""
# Check if TensorFlow detect GPU and setup the memory growth
setup_device()

# Call MTCNN function to create the detector
detector = MTCNN_setup()

# Open webcam using index 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera cannot be accesed.")
    exit()

print("Press 'Space' to take a photo or 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't receive frame. Exiting...")
        break

    # Show the webcam frame
    cv2.imshow('Webcam - Press Space to take a frame', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):# If 'Space' is pressed, capture and process the image

        output = process_face_image(frame, detector)

        # Create the results directory for webcam captures
        output_dir = 'results/webcam'
        os.makedirs(output_dir, exist_ok=True)

        # Show processed image
        cv2.imshow('Processed Image', output)

        # Save the processed image with an unique name
        timestamp = int(time.time())
        filename = f'webcam_{timestamp}.jpg'
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output)
        print(f"Webcam capture processed and saved as '{output_path}'")

    elif key == ord('q'):  # Press 'q' to quit
        break

# Free webcam and close windows
cap.release()
cv2.destroyAllWindows()
