import numpy as np
from scipy.interpolate import Rbf
from tqdm import tqdm


def transform_grid2contour(original_contour, deformed_contour, original_grid, test_grid, progress=''):
    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour,
                                            deformed_contour=deformed_contour)

    # Apply transformation to original contour to check whether deformation works as expected
    trafo_contour = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in original_contour
    ])

    # Deform original grid to coordinate system of deformed contour
    trafo_grid = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in tqdm(original_grid, desc="Transforming grid" + progress)
    ])

    trafo_ver_grid = np.array([
        [rbf_x(x, y), rbf_y(x, y)] for x, y in test_grid
    ])

    return trafo_grid, trafo_ver_grid, trafo_contour


# ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
def create_rbf_interpolators(original_contour, deformed_contour, function='linear', smooth=0.2):
    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    rbf_x = Rbf(x_orig, y_orig, x_deform, function=function, smooth=smooth)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function=function, smooth=smooth)

    return rbf_x, rbf_y


import numpy as np


def extend_grid(measurement_grids, x_extend, y_extend):
    """
    Extends the overall grid based on multiple measurement grids.
    """
    # Find global min/max across all grids
    all_points = np.vstack(measurement_grids)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Compute spacing separately for each grid
    x_spacings = []
    y_spacings = []

    for grid in measurement_grids:
        x_sorted = np.sort(np.unique(grid[:, 0].round(decimals=0)))
        y_sorted = np.sort(np.unique(grid[:, 1].round(decimals=0)))

        x_diff = np.diff(x_sorted)
        y_diff = np.diff(y_sorted)

        if len(x_diff) > 0:
            x_spacings.append(np.median(x_diff))  # Median x-spacing for this grid
        if len(y_diff) > 0:
            y_spacings.append(np.median(y_diff))  # Median y-spacing for this grid

    # Take the median across all grids to determine global spacing
    x_median_spacing = np.median(x_spacings) if x_spacings else 1
    y_median_spacing = np.median(y_spacings) if y_spacings else 1

    # Compute extension factor
    x_extend_factor = x_extend * (x_max - x_min)
    y_extend_factor = y_extend * (y_max - y_min)

    # Generate extended coordinate range
    x_new = np.arange(x_min - x_extend_factor, x_max + x_extend_factor + x_median_spacing, x_median_spacing)
    y_new = np.arange(y_min - y_extend_factor, y_max + y_extend_factor + y_median_spacing, y_median_spacing)

    # Create the extended meshgrid
    x, y = np.meshgrid(x_new, y_new)
    extended_grid_regular = np.stack([x, y], axis=-1)
    extended_grid_stack = np.vstack([extended_grid_regular[:, :, 0].ravel(), extended_grid_regular[:, :, 1].ravel()]).T
    extended_grid_shape = list(x.shape)

    return extended_grid_stack, extended_grid_shape

