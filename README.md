# ImageRetarget-python

ImageRetarget-python is a content-aware image retargeting project built using Python, OpenCV (`cv2`), NumPy, and Matplotlib. This project is indeed an improved version of the seam carving method. It leverages advanced techniques such as MiDaS for depth estimation and U-2-Net for saliency detection to perform content-aware image resizing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## Prerequisites

- Python 3.x
- `pip` for Python package management

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/ImageRetarget-python.git
   cd ImageRetarget-python
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Setup

### 1. MiDaS Depth Estimation

1. Clone the MiDaS repository:

   ```bash
   git clone https://github.com/isl-org/MiDaS.git
   ```

2. Copy all files from the MiDaS repository into the `MiDaS` directory of this project.

3. Replace the default `run.py` in MiDaS with the customized `run.py` provided in `/MiDas/run,py`

4. Download the MiDaS model weights as specified in the MiDaS repository and place them in the correct directory.

### 2. U-2-Net Saliency Detection

1. Clone the U-2-Net repository:

   ```bash
   git clone https://github.com/NathanUA/U-2-Net.git
   ```

2. Run the U-2-Net model to generate saliency maps for your input images. Make sure the saliency output directory matches the path defined in `/Functions/SaliencyFunction.py`.

### 3. Directory Configuration

1. Double-check that the output directory from U-2-Net matches the path specified in `/Functions/SaliencyFunction.py`. Adjust the path if necessary to ensure proper integration.

## Usage

Once you have set up the required repositories and directories, you can run the image retargeting code:

```bash
python main.py
```
The script will automatically use the outputs from the U-2-Net model and the MiDaS model to perform content-aware image retargeting.

You can also use the `main.ipynb` to better visualize the process.


## Output

- The final retargeted images will be saved in the `/Outputs` directory.
- Ensure that the output directory exists, or the script will create it automatically.

## Troubleshooting

- If you encounter issues with the U-2-Net outputs not being found, verify that the output directory in `/Functions/SaliencyFunction.py` matches the directory where U-2-Net is saving its results.
- Ensure that the MiDaS model weights are correctly downloaded and placed in the expected directory.
- Verify that the paths in the command line arguments are correct.

## Credits

- **MiDaS** for depth estimation: [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- **U-2-Net** for saliency detection: [U-2-Net GitHub](https://github.com/NathanUA/U-2-Net)
- **Improved Seam Carving with Forward Energy**: [Avik Das' Seam Carving with Forward Energy](https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html)
- **Content-Aware Image Resizing: An Improved and Shadow-Preserving Seam Carving Method**: [DOI: 10.1016/j.sigpro.2018.09.037](https://doi.org/10.1016/j.sigpro.2018.09.037)
