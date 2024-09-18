"""Compute depth maps for the given image
"""

import torch
import numpy as np
import cv2

from torchvision.transforms import Compose
from MiDaS.midas.dpt_depth import DPTDepthModel
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(image, model_type="dpt_large", optimize=True):

    default_models = {
        "midas_v21_small": "./MiDaS/weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "./MiDaS/weights/midas_v21-f6b98070.pt",
        "dpt_large": "./MiDaS/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "./MiDaS/weights/dpt_hybrid-midas-501f0c75.pt",
    }

    model_path = default_models[model_type]

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\tdeivce: {device}")

    # load network
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(
            model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True:

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    img = image / 255.0
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth_min = prediction.min()
    depth_max = prediction.max()

    bits = 2

    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=prediction.type)

    print("\tDone")

    return out.astype(np.float32)
