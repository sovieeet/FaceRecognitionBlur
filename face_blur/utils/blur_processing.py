import cv2
import numpy as np


def apply_blur_and_mask(output, mask):
    """
    This function applies a blur filter to the image and combines it with a mask.
    """
    # Apply blur filter to the entire image
    blurred_image = cv2.GaussianBlur(output, (115, 115), 20)
    """
    You can adjust the kernel size and the sigma value to change the blur effect for distinct images because every image could need a
    specify setting to blur correctly. Remember the kernel size must be an odd number.
    EX: blurred_image = cv2.GaussianBlur(output, (65, 65), 10) (less blur)
    EX2: blurred_image = cv2.GaussianBlur(output, (85, 85), 15) (intermediate blur)
    """
    # Apply an blur to the mask to smooth the edges
    mask = cv2.GaussianBlur(mask, (65, 65), 15)
    """
    You can adjust the kernel size to change the smooth effect for distinct images because every image could need a
    specify setting to blur correctly. Remember the kernel size must be an odd number.
    EX: mask = cv2.GaussianBlur(mask, (31, 31), 5) (less smooth)
    EX2: mask = cv2.GaussianBlur(mask, (65, 65), 15) (intermediate smooth)
    EX3: mask = cv2.GaussianBlur(mask, (91, 91), 30) (more smooth)
    """
    # Normalice the mask so the values are between 0 and 1
    mask_normalized = mask / 255.0

    # Create a 3 channel mask
    mask_3_channels = cv2.merge([mask_normalized, mask_normalized, mask_normalized])

    # Combine the blur image and the original using the smoothed mask
    output = (blurred_image * mask_3_channels + output * (1 - mask_3_channels)).astype(
        np.uint8
    )

    return output
