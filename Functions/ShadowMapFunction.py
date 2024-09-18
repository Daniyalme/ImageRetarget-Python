import cv2 as cv
import numpy as np


def shadow_map_function(image):

    YCrCb_img = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)

    y = YCrCb_img[:, :, 0]

    # Mask for shadow pixels
    shadows = np.zeros_like(y)

    avg_y_global = np.average(y)

    # Initial Value of Sliding Window
    B = 81

    while B >= 3:
        # Calculating the average intensity in sliding window
        kernel = np.ones((B, B), np.float32) / (B * B)
        avg_y_local = cv.filter2D(y, -1, kernel, borderType=cv.BORDER_REFLECT_101)

        # Detecting New Shadows
        new_mask = np.bitwise_or((y < (0.7 * avg_y_local)), y < (0.6 * avg_y_global))

        shadows = np.bitwise_or(shadows, new_mask)

        # Reducing the size of scrolling window
        B = B - 16

    # Applying Median Filter on the shadow map
    shadows = cv.medianBlur(shadows, ksize=3)

    noise = np.random.rand(image.shape[0], image.shape[1]) * 0.1

    return shadows.astype(np.float32) + noise
