import numpy as np
import cv2 as cv
import Functions.utils as utils


def intersection_function(energy_map, edges, seam, show=False, ksize=5, weights=[1, 1]):
    """
    Computes the intersection of the energy map and the edge map to refine the seam location
    for content-aware image retargeting.

    Parameters:
    - energy_map (2D array): Represents the energy of each pixel in the image. Higher values indicate higher importance in preserving those pixels.
    - edge_map (2D array): Highlights the edges in the image, typically obtained through edge detection techniques.
    - seam_location (list or array): Indicates the path of the seam to be removed or preserved.
    - show (bool): If True, the function displays the resulting image with the intersected seam overlaid.
    - ksize (int): The size of the Gaussian blur kernel to smooth the intersection mask.
    - weights (tuple of two floats): Coefficients used for combining the seam mask and intersection mask.

    Returns:
    - modified_energy_map (2D array): The energy map modified by incorporating the intersection of the seam and edge maps.
    """

    mask = np.zeros_like(energy_map)
    seam_mask = np.zeros_like(energy_map)

    h, w = energy_map.shape
    for row in range(h):
        # Marking the Seam in seam_mask
        seam_mask[row, seam[row]] = 1
        if edges[row, seam[row]] > 0:
            # Marking the Intesection of Edges with Seam
            mask[row, seam[row]] = 1

    # Blurring Blurring the Mask
    intersection_mask = cv.GaussianBlur(mask, (ksize, ksize), 0)
    seam_mask = cv.GaussianBlur(seam_mask, (ksize, ksize), 0)

    if show:
        utils.imshow([seam_mask, intersection_mask, energy_map, edges])

    return energy_map + (weights[0] * intersection_mask) + (weights[1] * seam_mask)
