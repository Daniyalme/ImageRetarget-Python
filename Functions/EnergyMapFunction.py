import cv2
import numpy as np


def energy_map_function(
    gradient_map,
    depth_map,
    saliency_map,
    shadow_map,
    edge_map,
    mode="add",
    weights=[1, 1, 1, 1, 1],
):

    g_map = gradient_map.copy().astype(np.float64)
    d_map = depth_map.copy().astype(np.float64)
    s_map = saliency_map.copy().astype(np.float64)
    sh_map = shadow_map.copy().astype(np.float64)
    e_map = edge_map.copy().astype(np.float64)

    # Normalizing Each map
    g_map = (g_map - np.min(g_map)) / (np.max(g_map) - np.min(g_map))
    d_map = (d_map - np.min(d_map)) / (np.max(d_map) - np.min(d_map))
    s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    sh_map = (sh_map - np.min(sh_map)) / (np.max(sh_map) - np.min(sh_map))
    e_map = (e_map - np.min(e_map)) / (np.max(e_map) - np.min(e_map))

    # Creating the Energy Map
    if mode == "add":
        energy_map = (
            (weights[0] * g_map)
            + (weights[1] * d_map)
            + (weights[2] * s_map)
            + (weights[3] * sh_map)
            + (weights[4] * e_map)
        ) / 5
        return energy_map

    if mode == "mul":
        energy_map = g_map * s_map * sh_map
        return energy_map
