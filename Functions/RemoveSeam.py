import numpy as np


def remove_seam(image, seam):

    if image.ndim == 2:
        img_h, img_w = image.shape
        new_img = np.zeros((img_h, img_w - 1))
        for row, col in enumerate(seam):
            new_img[row, :col] = image[row, :col]
            new_img[row, col:] = image[row, col + 1 :]

    if image.ndim == 3:
        img_h, img_w, ch = image.shape
        new_img = np.zeros((img_h, img_w - 1, ch))

        for row, col in enumerate(seam):
            new_img[row, :col, :] = image[row, :col, :]
            new_img[row, col:, :] = image[row, col + 1 :, :]

    return new_img
