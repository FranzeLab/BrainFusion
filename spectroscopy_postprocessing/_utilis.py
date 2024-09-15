import numpy as np
import matplotlib.path as mpath


def bin_data(data, bin_factor):
    """Bin the data by the given bin factor, handling non-divisible dimensions by trimming."""
    x_dim, y_dim = data.shape

    # Trim the dimensions to be divisible by bin_factor
    x_trim = x_dim - (x_dim % bin_factor)
    y_trim = y_dim - (y_dim % bin_factor)

    trimmed_data = data[:x_trim, :y_trim]

    # Reshape and bin
    binned_data = trimmed_data.reshape((x_trim // bin_factor, bin_factor,
                                        y_trim // bin_factor, bin_factor)).mean(axis=(1, 3))

    return binned_data


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    points = np.vstack((grid[:, :, 0].ravel(), grid[:, :, 1].ravel())).T
    mask_points = path.contains_points(points)

    # Reshape the result back to the shape of the mask
    mask = mask_points.reshape(grid[:, :, 0].shape)

    return mask


def center_contour(contour):
    # Calculate the centroid (center of mass) of the contour
    centroid = np.mean(contour, axis=0)

    # Translate all points so the centroid is at the origin
    centered_contour = contour - centroid

    return centered_contour, centroid


def create_data_grid(data):
    # Generate a grid covering the area for both contours
    x = np.arange(0, len(data[0, :]), 1)
    y = np.arange(0, len(data[:, 0]), 1)

    X, Y = np.meshgrid(x, y)  # Create grid of coordinates
    data_grid = np.stack((X, Y), axis=-1)

    return data_grid
