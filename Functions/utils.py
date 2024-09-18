import cv2 as cv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def imshow(images, titles=[], size=5):

    total_images = len(images)

    if total_images < 1:
        return
    if total_images == 1:
        if len(images[0].shape) == 2:
            plt.imshow(images[0], cmap="gray")
        else:
            plt.imshow(images[0])
        plt.axis("off")
        if len(titles) > 0:
            plt.title(titles[0])
        plt.show()

    else:
        plt.figure(figsize=(total_images * size, size))
        for i in range(total_images):
            plt.subplot(1, total_images, i + 1)
            if len(images[i].shape) == 2:
                plt.imshow(images[i], cmap="gray")
            else:
                plt.imshow(images[i])
            plt.axis("off")
            if i < len(titles):
                plt.title(titles[i])

        plt.show()


def highlight_seam(image, seam, color=(255, 0, 0)):

    res = image.copy()

    for i in range(image.shape[0]):
        res[i, seam[i]] = color

    return res


def normalize_img(image):

    img = image.copy().astype(np.float64)

    return (img - np.min(img)) / (np.max(img) - np.min(img))
