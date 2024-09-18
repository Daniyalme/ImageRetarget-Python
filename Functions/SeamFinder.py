import numpy as np


def seam_finder(forward_energy_map, col):

    seam = []

    # Last Row Location
    seam.append(col)

    h, w = forward_energy_map.shape

    current_col = col

    for i in reversed(range(0, h - 1)):
        left_col = max(0, current_col - 1)
        right_col = min(w - 1, current_col + 1)

        eligible_parents = np.arange(left_col, right_col + 1)

        parent = eligible_parents[np.argmin(forward_energy_map[i, left_col:right_col])]

        current_col = parent

        seam.append(current_col)

    return list(reversed(seam))
