# Default Imports
import os
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

# Local Imports
import Functions.utils as util
from Functions.RemoveSeam import remove_seam
from Functions.IntersectionFunction import intersection_function
from Functions.SeamFinder import seam_finder
from Functions.EdgeMapFunction import edge_map_function
from Functions.GradientMapFunction import gradient_map_function
from Functions.SaliencyMapFunction import saliency_map_function
from Functions.DepthMapFunction import depth_map_function
from Functions.ShadowMapFunction import shadow_map_function
from Functions.EnergyMapFunction import energy_map_function
from Functions.ForwardEnergyCalculationFunction import forward_energy_function

# Default Variables
input_path = "./Inputs/"
output_path = "./Outputs/"
reduction_factors = [0.75, 0.75, 0.5, 0.5, 0.5]
sass = 3  # Seam Array Selection Size

# Reading Images
images_list = os.listdir(input_path)
images_path = [input_path + item for item in images_list]

# Iterating Through the Images
for idx, image_path in enumerate(images_path):
    image_name = images_list[idx][:-4]
    I = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    I_org = I.copy()

    target_size = I.shape[1] * reduction_factors[idx]

    print("=" * 50)
    print(f"Working on {images_path}")
    print(f"Current Shape: {I.shape}")
    print(f"Target Shape: {I.shape[0],int(target_size),I.shape[2]}")
    print()

    # Creating the Edge Map of the Image
    edge_map = edge_map_function(I)

    # Creating the Gradient Map
    gradient_map = gradient_map_function(I)

    # Creating the Shadow Map
    shadow_map = shadow_map_function(I)

    # Creating the Shadow Map
    saliency_map = saliency_map_function(image_name)

    # Creating the Depth Map
    depth_map = depth_map_function(I)

    # Creating the Energy Map from (Gradient, Depth, Saliency, Shadow)
    energy_map = energy_map_function(
        gradient_map,
        depth_map,
        saliency_map,
        shadow_map,
        edge_map,
        mode="add",
        weights=[1, 2, 1, 0, 1],
    )

    # util.imshow(
    #     [gradient_map, saliency_map, shadow_map, depth_map, energy_map],
    #     ["Gradient Map", "Saliency Map", "Shadow Map", "Depth Map", "Energy Map"],
    # )

    total_seam_count = I.shape[1] - target_size
    seam_count = 0

    print("Seam Removal Process")

    # Applying Improved Seam Carving till width/height is reduced by [reduction_factor]

    while I.shape[1] > target_size:

        print(
            f"\tProgress: [ {(seam_count+1)/total_seam_count*100:.2f}% ]   ", end="\r"
        )

        # Calculating the Forward Energy Map for the Image
        forward_map = forward_energy_function(energy_map)

        # Selecting Top [seam_array_selection_size] from Forward Energy Map
        seam_energy = forward_map[-1, :]
        seams_loc = np.argsort(seam_energy)
        seams_loc = seams_loc[:sass]
        seams = [seam_finder(forward_map, seams_loc[i]) for i in range(sass)]

        # Iterating through the seam array

        best_score = -5

        for seam in seams:

            I_seam = remove_seam(I.copy(), seam)

            # Resizing the original image to match the size of the carved image
            I_resized = cv.resize(I, (I_seam.shape[1], I_seam.shape[0]), cv.INTER_CUBIC)

            # Calculating the SSIM Score for the selected seam
            score = ssim(I_resized, I_seam, channel_axis=-1, data_range=255)
            # score = Custom_Method(I,I_seam)

            # Updating the Best Score
            if score > best_score:
                best_score = score
                final_seam = seam

        # Removing the Seam with best score
        # Checking for intesection with Edge Map and Applying Filter to Intersections
        energy_map = intersection_function(
            energy_map, edge_map, final_seam, show=False, ksize=15, weights=[1.2, 1.2]
        )

        # Highlighting the seams to Green
        highlighted_seam = I.copy()
        for seam in seams:
            highlighted_seam = util.highlight_seam(highlighted_seam, seam, (0, 255, 0))

        # Highlighting the Chosen seam to Red
        highlighted_seam = util.highlight_seam(highlighted_seam, final_seam)

        # Showing the Reuslt
        img1 = util.normalize_img(highlighted_seam)
        img2 = util.normalize_img(energy_map.copy())
        img2 = cv.merge([img2, img2, img2])

        result = (255 * cv.hconcat([img1, img2])).astype(np.uint8)

        cv.imshow("Result", cv.cvtColor(result, cv.COLOR_RGB2BGR))
        cv.waitKey(1)

        # Removing the Seam from Image
        I = remove_seam(I, final_seam)

        # Removing the Seam from Energy Map
        energy_map = remove_seam(energy_map, final_seam)
        edge_map = remove_seam(edge_map, final_seam)

        seam_count += 1

    # util.imshow([I_org, I.astype(np.uint8)], ["Original Image", "Seam Carved Image"])

    cv.imwrite(
        output_path + image_name + ".png",
        cv.cvtColor(I.astype(np.uint8), cv.COLOR_RGB2BGR),
    )
    cv.destroyAllWindows()
