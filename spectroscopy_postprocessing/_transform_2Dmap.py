import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, griddata
from _utilis import mask_contour, center_contour


def create_rbf_interpolators(original_contour, deformed_contour):
    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    rbf_x = Rbf(x_orig, y_orig, x_deform, function='thin_plate')
    rbf_y = Rbf(x_orig, y_orig, y_deform, function='thin_plate')

    return rbf_x, rbf_y


def evaluate_transformation(rbf_x, rbf_y, grid_points):
    grid_x, grid_y = grid_points[:, 0], grid_points[:, 1]
    trafo_grid_x = rbf_x(grid_x, grid_y)
    trafo_grid_y = rbf_y(grid_x, grid_y)

    return trafo_grid_x, trafo_grid_y


def plot_cont_func(original_contour, deformed_contour, trafo_contour, data_map, grid_points, data_map_trafo, grid_points_trafo):
    # Create raveled grid points
    grid_points_rav = np.vstack([grid_points[:, :, 0].ravel(), grid_points[:, :, 1].ravel()]).T
    grid_points_trafo_rav = np.vstack([grid_points_trafo[:, :, 0].ravel(), grid_points_trafo[:, :, 1].ravel()]).T

    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    original_mask = mask_contour(original_contour, grid_points)
    deformed_mask = mask_contour(deformed_contour, grid_points_trafo)
    axes[0].scatter(grid_points_rav[:, 0],
                    grid_points_rav[:, 1],
                    c=data_map,
                    cmap='hot',
                    s=1,
                    label='Original Grid',
                    alpha=1)
    axes[0].scatter(grid_points_trafo_rav[:, 0],
                    grid_points_trafo_rav[:, 1],
                    c=data_map,
                    cmap='viridis',
                    s=1,
                    label='Transformed Grid',
                    alpha=0)

    axes[0].legend()
    axes[0].set_title('Function transformation between deformed domains')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    original_mask = mask_contour(original_contour, grid_points)
    deformed_mask = mask_contour(deformed_contour, grid_points)
    axes[1].scatter(grid_points_rav[:, 0],
                    grid_points_rav[:, 1],
                    c=data_map,
                    cmap='hot',
                    s=1,
                    label='Original Grid',
                    alpha=0)
    axes[1].scatter(grid_points_rav[:, 0],
                    grid_points_rav[:, 1],
                    c=data_map_trafo,
                    cmap='viridis',
                    s=1,
                    label='Transformed Grid',
                    alpha=1)

    axes[1].legend()
    axes[1].set_title('Function transformation + interpolation to regular grid')
    axes[1].grid()
    axes[1].axis('equal')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def transform_map2contour(original_contour, deformed_contour, data_map, original_grid):
    # Shift centroids of contours to coordinate systems origin
    original_contour_shift, original_centroid = center_contour(original_contour)
    deformed_contour_shift, deformed_centroid = center_contour(deformed_contour)

    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour_shift,
                                            deformed_contour=deformed_contour_shift)

    # Transform original contour to the coordinate system of the deformed contour
    trafo_cont_x, trafo_cont_y = evaluate_transformation(rbf_x, rbf_y, original_contour_shift)
    trafo_matched_contour = np.vstack([trafo_cont_x, trafo_cont_y]).T
    trafo_matched_contour = trafo_matched_contour + deformed_centroid

    # Reshape original grid to a Nx2 array
    original_grid_points = np.vstack([original_grid[:, :, 0].ravel(), original_grid[:, :, 1].ravel()]).T

    # Transform original grid to the coordinate system defined by the average contour
    trafo_grid_x, trafo_grid_y = evaluate_transformation(rbf_x, rbf_y, original_grid_points - original_centroid)
    trafo_grid_points = np.vstack([trafo_grid_x, trafo_grid_y]).T
    trafo_grid_points = trafo_grid_points + deformed_centroid

    # Interpolate data_map from the regular grid to the transformed grid
    data_map_trafo = griddata(trafo_grid_points,
                              data_map.ravel(),
                              original_grid_points,
                              method='linear')
    trafo_grid = np.stack(
        (np.reshape(trafo_grid_points[:, 0], original_grid[:, :, 0].shape),
         np.reshape(trafo_grid_points[:, 1], original_grid[:, :, 1].shape)), axis=-1)

    # Reshape data_map_trafo back to the original data dimensions
    data_map_trafo = data_map_trafo.reshape(original_grid[:, :, 0].shape)

    return data_map_trafo, trafo_grid, trafo_matched_contour
