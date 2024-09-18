import cv2 as cv

# Canny Operator Configuration
KERNEL_SIZE = 3
USE_OPTIMIZED_FORMULA = True
MIN_VAL = 100
MAX_VAL = 200


def edge_map_function(image):
    # Converting the RGB Image to Gray
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Extracting the EdgeMap Using Canny
    edge_map = cv.Canny(
        gray_image,
        MIN_VAL,
        MAX_VAL,
        L2gradient=USE_OPTIMIZED_FORMULA,
        apertureSize=KERNEL_SIZE,
    )

    # Dialating Image using 3x3 rectangle SE
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    return cv.dilate(edge_map, se)
