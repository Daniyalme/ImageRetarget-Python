# Image Retargeting Project

This directory contains various Python scripts used for content-aware image retargeting. Each script performs a specific task related to image processing, such as calculating energy maps, finding seams, and adjusting images using seam carving techniques. Below is a detailed description of each file:

## Files and Descriptions

- **DepthMapFunction.py**: Calculates the depth map of an image using the MiDaS depth estimation model. The depth map provides information about the relative distances of objects within the scene, which can be used to preserve important regions during image resizing.

- **EdgeMapFunction.py**: Computes the edge map of an image. This highlights the edges within the image, which are important features to retain during seam carving.

- **EnergyMapFunction.py**: Generates the final energy map of the image. The energy map indicates the importance of each pixel, guiding the seam carving process by assigning higher values to regions that should be preserved.

- **ForwardEnergyCalculationFunction.py**: Implements forward energy map calculation using the method described [here](https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html). Forward energy calculation improves seam carving by considering the impact of removing seams on the surrounding pixels, leading to more natural-looking resized images.

- **GradientMapFunction.py**: Calculates the gradient map of the image. The gradient map measures the intensity changes in the image, often used as a basis for calculating the energy map.

- **IntersectionFunction.py**: Finds the intersection between a seam and the edges of an image. It increases the energy in the energy map within a specified range around the intersection to preserve crucial features during seam removal.

- **RemoveSeam.py**: Removes a seam (specifically a vertical seam) from the image. Seam removal is the core operation in seam carving, where a path of low-energy pixels is removed to resize the image.

- **SaliencyMapFunction.py**: Computes the saliency map for an image using the U-2-Net model for saliency detection. The saliency map identifies the most visually significant regions in the image, helping to ensure that important features are preserved during resizing.

- **SeamFinder.py**: Finds the optimal seam in the image based on the provided energy map. This seam represents the path of pixels that have the lowest cumulative energy and can be removed or preserved.

- **ShadowMapFunction.py**: Calculates the shadow map of the image, which is used to preserve shadows during seam carving. Preserving shadows helps maintain the visual coherence of the image when resized.

- **utils.py**: Contains utility functions used throughout the main codebase. These functions provide basic functionalities that support the primary image processing tasks.
