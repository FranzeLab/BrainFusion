import numpy as np


def apply_affine_transform(coords, affine):
    """
    Apply a 3x3 affine transformation to Nx2 coordinates.
    """
    coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N, 3)
    transformed = coords_hom @ affine.T  # (N, 3)
    return transformed[:, :2]  # drop homogeneous coordinate


def map_values_to_grid(grid_points, values, regular_grid):
    """
    Maps Nx1 values to an (H, W) matrix.
    """
    H, W = regular_grid.shape[:2]
    value_matrix = np.full((H, W), np.nan)

    # Create fast lookup: map grid coordinates to indices
    coord_to_index = {(x, y): (i, j)
                      for i in range(H)
                      for j in range(W)
                      for x, y in [tuple(regular_grid[i, j])]}

    for point, val in zip(grid_points, values):
        key = tuple(np.round(point, decimals=6))  # match precision with regular_grid
        if key in coord_to_index:
            i, j = coord_to_index[key]
            value_matrix[i, j] = val

    return value_matrix