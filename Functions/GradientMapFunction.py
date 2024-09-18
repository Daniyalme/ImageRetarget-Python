import cv2
import numpy as np


def gradient_map_function(image):
    # Converting RGB to Gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculating the Gradient in both direction
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Convert gradients to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Combine the gradients
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad
