import numpy as np
import matplotlib.path as mpath
from scipy.stats import binned_statistic_2d


def apply_affine_transform(coords, affine):
    """
    Apply a 3x3 affine transformation to Nx2 coordinates.
    """
    coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])  # (N, 3)
    transformed = coords_hom @ affine.T  # (N, 3)
    return transformed[:, :2]  # drop homogeneous coordinate


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


def bin_single_image_channel(img, bin_size, method='mean', crop=True):
    N, M = img.shape

    if crop:
        new_N = N - (N % bin_size)
        new_M = M - (M % bin_size)
        img = img[:new_N, :new_M]
    else:
        # Pad to make divisible
        pad_N = (bin_size - N % bin_size) % bin_size
        pad_M = (bin_size - M % bin_size) % bin_size
        img = np.pad(img, ((0, pad_N), (0, pad_M)), mode='constant')

    # Bin the image
    reshaped = img.reshape(img.shape[0] // bin_size, bin_size,
                           img.shape[1] // bin_size, bin_size)

    if method == 'mean':
        return reshaped.mean(axis=(1, 3))
    elif method == 'sum':
        return reshaped.sum(axis=(1, 3))
    elif method == 'max':
        return reshaped.max(axis=(1, 3))
    else:
        raise ValueError("Invalid method: choose 'mean', 'sum', or 'max'")


def transform_outline_for_binning(outline, bin_size, crop=False, original_shape=None):
    outline = np.asarray(outline, dtype=float)

    if crop and original_shape is not None:
        N, M = original_shape
        new_N = N - (N % bin_size)
        new_M = M - (M % bin_size)
        # Filter out points that are outside the cropped region
        keep = (outline[:, 0] < new_M) & (outline[:, 1] < new_N)
        outline = outline[keep]

    # Scale down coordinates
    outline = (outline - 0.5) / np.sqrt(bin_size) + 0.5
    return outline


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
