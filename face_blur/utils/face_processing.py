import cv2
import numpy as np
from .blur_processing import apply_blur_and_mask


# Function to rotate the image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# Function to create a mask for the detected faces
def create_face_mask(faces, image, mask):
    """
    This function creates a mask with white circles in the face positions.
    """
    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        x2, y2 = x + width, y + height
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Calculate the center and the radius of the circle
        center_x = x + width // 2
        center_y = y + height // 2
        radius = int(
            max(width, height) * 0.7
        )  # You can adjust this value if the circle is too big/small
        """
        Adjust the value of the radius to change the size of the circle
        EX: radius = int(max(width, height) * 0.3) (smaller circle)
        EX2: radius = int(max(width, height) * 0.7) (bigger circle)
        EX3: radius = int(max(width, height) * 1.0) (circle with the same size as the face)
        """

        # Draw a white circle in the mask at the face position
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    return mask


# Main function to process the image
def process_face_image(image, detector):
    # Get the image dimensions
    (h, w) = image.shape[:2]

    # Create an empty mask for all detected faces
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

    # Detect faces in the image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    # Generate the mask for all detected faces
    if faces:  # Only proceed if faces are detected
        mask = create_face_mask(faces, image, mask)

        # cv2.imshow('Mask', mask) # Uncomment this line to show the mask in the image
        # cv2.waitKey(0)

        # Apply blur and mask using the imported function
        output = apply_blur_and_mask(image, mask)

        return output
    else:
        print("No faces detected in the image.")
        return image  # If no faces are detected, return the original image
