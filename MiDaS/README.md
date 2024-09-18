## MiDaS Setup Instructions

To use the depth estimation features provided in this project, you need to set up the MiDaS model. Follow the steps below:

1. **Clone the MiDaS Repository**:
   Clone the official MiDaS repository into a folder named `MiDaS` within your project directory:
   ```bash
   git clone https://github.com/isl-org/MiDaS.git MiDaS

2. **Replace the `run.py` File**:
   This project provides a custom `run.py` file tailored to work with the content-aware image retargeting pipeline. Replace the `run.py` file in the MiDaS folder with the one provided in this project:
   - Copy the `run.py` file from this project and paste it into the `MiDaS` folder, overwriting the existing `run.py` file.

3. **Install MiDaS Dependencies**:
   Navigate to the `MiDaS` folder and install the required dependencies by running:
   ```bash
   cd MiDaS
   pip install -r requirements.txt
   ```

4. **Using MiDaS in the Project**:
   Once the MiDaS model is set up with the modified `run.py` file, the `DepthMapFunction.py` function will calculate the energy map using MiDaS depth estimation automatically.

## Usage

With the MiDaS model set up and the provided scripts, you only need to download the weights related to models mentioned in the description file [here](https://github.com/isl-org/MiDaS)
