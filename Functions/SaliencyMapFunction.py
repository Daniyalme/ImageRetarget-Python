import cv2 as cv


def saliency_map_function(image_name):
    default_u2_net_path = "./U-2-Net/test_data/u2net_results/"

    full_path = default_u2_net_path + image_name + ".png"

    return cv.cvtColor(cv.imread(full_path), cv.COLOR_BGR2GRAY)
