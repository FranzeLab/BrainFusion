import numpy as np
from scipy.interpolate import Rbf
from tqdm import tqdm
from scipy.spatial import cKDTree, distance_matrix
import numbers


def transform_grid2contour(original_contour, deformed_contour, original_grid, test_grid, smooth='auto', progress=''):
    # Create RBF interpolators (forward and inverse)
    rbf_x, rbf_y, rbf_x_inv, rbf_y_inv = create_rbf_interpolators(original_contour=original_contour,
                                                                  deformed_contour=deformed_contour,
                                                                  smooth=smooth)

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

    return trafo_grid, trafo_ver_grid, trafo_contour, rbf_x_inv, rbf_y_inv


# ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
# ToDo: Implement smooth bijective transformation to quickly invert for faster interpolation to regular grid
def create_rbf_interpolators(original_contour, deformed_contour, function='linear', smooth='auto'):
    if smooth == "auto":
        D = distance_matrix(original_contour, original_contour)
        np.fill_diagonal(D, np.inf)
        typical_spacing = np.min(D, axis=1).mean()
        smooth = 0.1 * typical_spacing
    elif isinstance(smooth, numbers.Number) is False:
        raise Exception('The msoothing parameter is not valid. Use "auto" or aa number!')

    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    # Define forward mapping from definition space to deformation space
    rbf_x = Rbf(x_orig, y_orig, x_deform, function=function, smooth=smooth)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function=function, smooth=smooth)

    # Approximate backward mapping from deformation space to definition space
    rbf_x_inv = Rbf(x_deform, y_deform, x_orig, function=function, smooth=smooth)
    rbf_y_inv = Rbf(x_deform, y_deform, y_orig, function=function, smooth=smooth)

    return rbf_x, rbf_y, rbf_x_inv, rbf_y_inv


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

    # Compute the median of unique absolute x and y distances to the 4 nearest neighbours for a set of 2D coordinates
    for grid in measurement_grids:
        tree = cKDTree(grid)
        dists, idxs = tree.query(grid, k=5)

        seen_pairs = set()
        dx_list = []
        dy_list = []

        for i in range(len(grid)):
            for j_idx in range(1, 5):  # skip self (index 0)
                j = idxs[i, j_idx]
                pair = tuple(sorted((i, j)))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    dx = abs(grid[i, 0] - grid[j, 0])
                    dy = abs(grid[i, 1] - grid[j, 1])
                    dx_list.append(dx)
                    dy_list.append(dy)

        median_dx = np.median(dx_list)
        median_dy = np.median(dy_list)

        x_spacings.append(median_dx)
        y_spacings.append(median_dy)

    # Take the median across all grids to determine global spacing
    x_median_spacing = np.median(x_spacings)
    y_median_spacing = np.median(y_spacings)

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
