import numpy as np
from scipy.interpolate import Rbf, griddata
from scipy.spatial import cKDTree
from _find_average_contour import match_contour_with_ellipse, rotate_coordinate_system


# ToDo: Implement new interpolation function scipy.interpolate.RBFInterpolator
def create_rbf_interpolators(original_contour, deformed_contour):
    x_orig, y_orig = original_contour[:, 0], original_contour[:, 1]
    x_deform, y_deform = deformed_contour[:, 0], deformed_contour[:, 1]

    rbf_x = Rbf(x_orig, y_orig, x_deform, function='linear', smooth=0.2)
    rbf_y = Rbf(x_orig, y_orig, y_deform, function='linear', smooth=0.2)

    return rbf_x, rbf_y


def evaluate_transformation(rbf_x, rbf_y, grid_points):
    grid_x, grid_y = grid_points[:, 0], grid_points[:, 1]
    trafo_grid_x = rbf_x(grid_x, grid_y)
    trafo_grid_y = rbf_y(grid_x, grid_y)

    return trafo_grid_x, trafo_grid_y


def extend_grid(regular_grid, x_extend, y_extend):
    # Extract spacing for x and y direction
    x_sorted = np.sort(np.unique(regular_grid[:, 0]))
    y_sorted = np.sort(np.unique(regular_grid[:, 1]))
    x_spacing = np.diff(x_sorted)
    y_spacing = np.diff(y_sorted)

    assert np.allclose(x_spacing, x_spacing[0]), print('Original grid is not regularly spaced in x-direction!')
    assert np.allclose(y_spacing, y_spacing[0]), print('Original grid is not regularly spaced in y-direction!')

    # Get minimum and maximum coordinate values in x and y
    x_min, y_min = np.min(regular_grid, axis=0)
    x_max, y_max = np.max(regular_grid, axis=0)

    # Generate new extended coordinates
    x_new = np.arange(x_min - x_extend, x_max + x_extend + x_spacing[0], x_spacing[0])
    y_new = np.arange(y_min - y_extend, y_max + y_extend + y_spacing[0], y_spacing[0])

    # Create the extended grid
    X, Y = np.meshgrid(x_new, y_new)
    extended_grid = np.stack([X, Y], axis=-1)

    return extended_grid


def nearest_neighbor_interp(grid, data, grid_extended, max_distance):
    # Perform interpolation using griddata
    interp_data = griddata(grid, data, grid_extended, method='nearest')

    # Find nearest neighbors between extended grid and irregular grid
    tree = cKDTree(grid)
    dist, _ = tree.query(grid_extended)

    # Set values to NaN where no nearby irregular grid point exists
    interp_data[dist > max_distance] = np.nan

    return interp_data


def transform_map2contour(original_contour, deformed_contour, bm_data, bm_metadata, bm_template_grid):
    # Shift centroids of contours to coordinate systems origin
    _, rotation_angle, org_center, def_center = match_contour_with_ellipse(original_contour, deformed_contour)
    original_contour_shift = original_contour - org_center
    deformed_contour_shift = deformed_contour - def_center

    # Rotate contour to maximize overlap of ellipse areas
    original_contour_rot = rotate_coordinate_system(original_contour_shift, rotation_angle, (0, 0))

    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour_rot,
                                            deformed_contour=deformed_contour_shift)

    # Transform original contour to the coordinate system of the deformed contour
    trafo_cont_x, trafo_cont_y = evaluate_transformation(rbf_x, rbf_y, original_contour_rot)
    trafo_contour = np.vstack([trafo_cont_x, trafo_cont_y]).T + def_center

    # Reshape original grid to a Nx2 array
    original_grid = bm_metadata['brillouin_grid'][:, :, 0, :]  # Use x-y grid of first z-slice
    original_grid_points = np.vstack([original_grid[:, :, 0].ravel(), original_grid[:, :, 1].ravel()]).T
    original_grid_points_rot = rotate_coordinate_system(original_grid_points - org_center, rotation_angle, (0, 0))

    # Transform original grid to the coordinate system defined by the average contour
    trafo_grid_x, trafo_grid_y = evaluate_transformation(rbf_x, rbf_y, original_grid_points_rot)
    trafo_grid_points = np.vstack([trafo_grid_x, trafo_grid_y]).T + def_center

    trafo_grid = np.stack(
        (np.reshape(trafo_grid_points[:, 0], original_grid[:, :, 0].shape),
         np.reshape(trafo_grid_points[:, 1], original_grid[:, :, 1].shape)), axis=-1)

    # Extend original grid of template contour to image boundaries
    x_extend, y_extend = 30, 30
    max_distance = 40
    bm_template_grid = bm_template_grid['brillouin_grid'][:, :, 0, :]
    bm_template_grid = np.vstack([bm_template_grid[:, :, 0].ravel(), bm_template_grid[:, :, 1].ravel()]).T
    extended_grid = extend_grid(bm_template_grid, x_extend, y_extend)
    extended_grid_points = np.vstack([extended_grid[:, :, 0].ravel(), extended_grid[:, :, 1].ravel()]).T

    bm_data_trafo = {}
    # Interpolate data maps from the regular grid to the transformed grid
    for key, data_map in bm_data.items():  # Access both key and data
        data_map_trafo = nearest_neighbor_interp(trafo_grid_points, data_map.ravel(), extended_grid_points, max_distance)

        # Reshape data_map_trafo back to the original data dimensions and store it in the new dictionary
        bm_data_trafo[key + '_trafo'] = data_map_trafo.reshape(extended_grid[:, :, 0].shape)

    return bm_data_trafo, trafo_grid, extended_grid, trafo_contour


def transform_map2contour_afm(original_contour, deformed_contour, data, grid):
    # Shift centroids of contours to coordinate systems origin
    _, rotation_angle, org_center, def_center = match_contour_with_ellipse(original_contour, deformed_contour)
    original_contour_shift = original_contour - org_center
    deformed_contour_shift = deformed_contour - def_center

    # Rotate contour to maximize overlap of ellipse areas
    original_contour_rot = rotate_coordinate_system(original_contour_shift, rotation_angle, (0, 0))

    # Create RBF interpolators
    rbf_x, rbf_y = create_rbf_interpolators(original_contour=original_contour_rot,
                                            deformed_contour=deformed_contour_shift)

    # Transform original contour to the coordinate system of the deformed contour
    trafo_cont_x, trafo_cont_y = evaluate_transformation(rbf_x, rbf_y, original_contour_rot)
    trafo_contour = np.vstack([trafo_cont_x, trafo_cont_y]).T + def_center

    # Reshape original grid to a Nx2 array
    original_grid_points = grid
    original_grid_points_rot = rotate_coordinate_system(original_grid_points - org_center, rotation_angle, (0, 0))

    # Transform original grid to the coordinate system defined by the average contour
    trafo_grid_x, trafo_grid_y = evaluate_transformation(rbf_x, rbf_y, original_grid_points_rot)
    trafo_grid_points = np.vstack([trafo_grid_x, trafo_grid_y]).T + def_center

    data_map_trafo = griddata(trafo_grid_points,
                              data.ravel(),
                              original_grid_points,
                              method='linear')

    return data_map_trafo, trafo_grid_points, trafo_contour
