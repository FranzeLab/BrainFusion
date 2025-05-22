import numpy as np
import matplotlib.path as mpath
from scipy.stats import binned_statistic_2d


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    mask = path.contains_points(grid)

    return mask


def regular_grid_on_contour(contour, axis_points=50):
    # Get min and max x, y coordinates
    min_x, min_y = np.min(contour, axis=0)
    max_x, max_y = np.max(contour, axis=0)

    # Generate linearly spaced grid points
    x_values = np.linspace(min_x, max_x, axis_points)
    y_values = np.linspace(min_y, max_y, axis_points)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    return grid


def project_brillouin_dataset(bm_data, bm_metadata, br_intensity_threshold=15):
    bm_data_proj = {}
    if 'brillouin_peak_intensity' in bm_data and 'brillouin_shift_f' in bm_data:
        # Filter out invalid peaks
        mask_peak = bm_data['brillouin_peak_intensity'] > br_intensity_threshold
        # Filter out water shifts
        mask_shift = (4.4 < bm_data['brillouin_shift_f']) & (bm_data['brillouin_shift_f'] < 10.0)

        # Check data distribution visually
        """
        import matplotlib.pyplot as plt
        cumulative_percentage = np.linspace(0, 100, len(sorted_data))
        plt.plot(sorted_data, cumulative_percentage, color='blue', linewidth=2)
        plt.title("Cumulative distribution")
        plt.xlabel("Brillouin shift (GHz)")
        plt.ylabel("Cumulative Percentage (%)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
        """

        mask = mask_peak & mask_shift & (0 < bm_data['brillouin_peak_fwhm_f']) & (bm_data['brillouin_peak_fwhm_f'] < 5)
    else:
        mask = True
    for key, value in bm_data.items():
        new_value = value.copy()  # Copy the original data to avoid modifying it

        # For distribution analysis
        sorted_data = np.sort(new_value.flatten())
        bm_data_proj[key + '_distribution'] = sorted_data  # Store the sorted distribution

        if key == 'brillouin_peak_intensity':
            continue

        new_value = np.where(mask, new_value, np.nan)

        proj_value = np.nanmedian(new_value, axis=-1).ravel()

        bm_data_proj[key + '_proj'] = proj_value  # Store the projection

    bm_grid_proj = bm_metadata['brillouin_grid'][:, :, 0, :2]  # Use x,y grid of first z-slice
    bm_grid_proj = np.column_stack([bm_grid_proj[:, :, 0].ravel(), bm_grid_proj[:, :, 1].ravel()])

    return bm_data_proj, bm_grid_proj


import numpy as np
from scipy.stats import binned_statistic_2d


def bin_single_image_channel(values, pixel_grid, bin_size=10):
    """
    Spatially bin an image by averaging values in 2D bins.
    """
    pixel_grid = np.asarray(pixel_grid)
    values = np.asarray(values).flatten()

    x, y = pixel_grid[:, 0], pixel_grid[:, 1]
    if len(values) != len(x):
        raise ValueError("Length of values must match number of coordinates in pixel_grid.")

    # Define bin edges
    x_bins = np.arange(x.min(), x.max() + bin_size, bin_size)
    y_bins = np.arange(y.min(), y.max() + bin_size, bin_size)

    # Bin the data
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        x, y, values, statistic='mean', bins=[x_bins, y_bins]
    )

    # Compute bin centres
    x_centres = (x_edges[:-1] + x_edges[1:]) / 2
    y_centres = (y_edges[:-1] + y_edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(x_centres, y_centres, indexing='ij')
    binned_grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

    # Flatten and filter out NaNs (empty bins)
    binned_values = stat.flatten()
    valid = ~np.isnan(binned_values)

    return binned_values[valid], binned_grid[valid]


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
