import numpy as np


def min_pooling_3x1(array):

    length = len(array)

    padded_array = np.pad(array, (1, 1), mode="edge")
    arr_2d = np.zeros((3, length))

    arr_2d[0] = padded_array[:length]  # Left
    arr_2d[1] = padded_array[1 : length + 1]  # Top
    arr_2d[2] = padded_array[2:]  # Right

    return np.min(arr_2d, axis=0)


def forward_energy_function(energy_map):

    h, w = energy_map.shape

    forward_energy_map = np.zeros_like(energy_map)
    forward_energy_map[0, :] = energy_map[0, :]

    for i in range(1, h):
        forward_energy_map[i] = energy_map[i] + min_pooling_3x1(
            forward_energy_map[i - 1]
        )

    return forward_energy_map
