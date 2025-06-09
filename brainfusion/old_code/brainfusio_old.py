import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

from brainfusion._match_contours import interpolate_contour, align_contours, boundary_match_contours
from brainfusion._average_contours import find_average_contour
from brainfusion._transform_2Dmap import extend_grid, transform_grid2contour
from brainfusion._interpolation import nearest_neighbour_interp, fit_coordinates_gmm
from brainfusion._utils import regular_grid_on_bbox


def fuse_measurement_datasets(base_path, template, grids, image_dims, scales, datasets, contours, points, axes, filenames,
                              bg_images, contour_interp_n=200, clustering='Mean', outline_averaging='star_domain',
                              pullback=False, smooth='auto', curvature=0.5, **kwargs):
    """
    Use this function to match tissue boundary outlines with measurement grids defined in an enclosed area. Boundaries
    can be matched using a given template or by calculating average tissue shape. The original measurement grids are
    then transformed to the template shape and together with the corresponding data averaged by interpolation to a
    regular grid or clustering using a Gaussian Mixture Model.
    """
    print(f'Processing folder: {base_path}.')

    # Close all contours
    closed_contours = []
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])  # Append the first point at the end
        closed_contours.append(contour)

    # Interpolate all contours to same length
    contours_interp = []
    for contour in closed_contours:
        contours_interp.append(interpolate_contour(contour, num_points=contour_interp_n))

    # Align contours and grids using affine transformations
    template_index = 0  # Must be 0
    aligned_contours, aligned_grids, affine_matrices = align_contours(contours_interp,
                                                                      grids,
                                                                      rot_axes=axes,
                                                                      init_points=points,
                                                                      template_index=template_index,
                                                                      fit_routine='ellipse')

    if template == 'average':
        # Calculate average contour and add as first element in list
        template_contour, errors = find_average_contour(aligned_contours, average=outline_averaging,
                                                        star_bins=contour_interp_n, error_metric='frechet')
        aligned_contours.insert(template_index, template_contour)
        aligned_grids.insert(template_index, np.empty((1, 2)))
        datasets.insert(template_index, np.empty((1, 2)))
    elif template == 'first_element':
        pass
    else:
        raise ValueError(f'Choice of template: {template} is not implemented!')

    # Match boundaries using DTW
    dtw_contours, dtw_template_contours = boundary_match_contours(aligned_contours, template_index=template_index,
                                                                  curvature=curvature)

    # Split template data from rest
    template_grid = aligned_grids[0]
    measurement_grids = aligned_grids[1:]

    template_dataset = datasets[0]
    measurement_datasets = datasets[1:]

    template_contours = dtw_template_contours[1:]
    measurement_contours = dtw_contours[1:]

    # Initialise grid to interpolate transformed data on
    extended_grid, extended_grid_shape = extend_grid(measurement_grids, 0.1, 0.1)

    # Create regular test grids for verifying correct transformation
    verification_grids = [regular_grid_on_bbox(c) for c in measurement_contours]

    # Using boundary matched contours to model plane transformation with RBF interpolation
    trafo_data_maps, trafo_grids, trafo_contours, trafo_ver_grids = [], [], [], []
    for index, contour in enumerate(measurement_contours):
        trafo_grid, trafo_ver_grid, trafo_contour, rbf_x_inv, rbf_y_inv = transform_grid2contour(
            measurement_contours[index],
            template_contours[index],
            measurement_grids[index],
            verification_grids[index],
            smooth=smooth,
            progress=f' {index + 1} out of {len(dtw_contours) - 1}')

        # Interpolate maps from deformed grids to common regular grid
        trafo_data = {}

        if pullback is True:  # Faster interpolation but introduces some heavy errors
            # Pull back extended grid to definition space
            x_pull = rbf_x_inv(extended_grid[:, 0], extended_grid[:, 1])
            y_pull = rbf_y_inv(extended_grid[:, 0], extended_grid[:, 1])
            points_pulled = np.vstack([y_pull, x_pull]).T  # (y, x) for RegularGridInterpolator

            # Extract unique axes from measurement grid
            measurement_reg = (measurement_grids[index].reshape(image_dims[index][0], image_dims[index][1], 2))
            x_axis = measurement_reg[0, :, 0]
            y_axis = measurement_reg[:, 0, 1]

            for key, data_map in measurement_datasets[index].items():
                # Reshape data map to 2D grid (y rows, x cols)
                data_reg = data_map.reshape(image_dims[index][0], image_dims[index][1])

                # Create interpolator
                interp_func = RegularGridInterpolator(
                    (y_axis, x_axis),  # note the tuple of coordinate arrays
                    data_reg,
                    method='linear',
                    bounds_error=True,
                    fill_value=np.nan
                )

                # Interpolate pulled-back points
                trafo_data[key] = interp_func(points_pulled)
        else:
            for key, data_map in measurement_datasets[index].items():
                trafo_data[key] = nearest_neighbour_interp(trafo_grid, data_map.ravel(), extended_grid, method='nearest', unique=False)
                #trafo_data[key] = griddata(points=trafo_grid, values=data_map.ravel(), xi=extended_grid, method='nearest')  # ToDo: Remove old code

        trafo_data_maps.append(trafo_data)
        trafo_grids.append(trafo_grid)
        trafo_ver_grids.append(trafo_ver_grid)
        trafo_contours.append(trafo_contour)

    # Average transformed datasets
    if clustering == 'GMM':
        # Cluster data points using Gaussian Mixture Model and average using median
        gmm_grid, gmm_dict, avg_data = fit_coordinates_gmm(trafo_grids,
                                                           measurement_datasets,
                                                           trafo_data_maps,
                                                           same_maps=True,
                                                           num_components='mean')
    elif clustering in ["Sum", "Median", "Mean"]:
        # Compute clustered array for each key
        keys = trafo_data_maps[0].keys()
        projection = None
        if clustering == "Sum":
            projection = np.nansum
        elif clustering == "Median":
            projection = np.nanmedian
        elif clustering == "Mean":
            projection = np.nanmean

        avg_data = {
            key: projection(np.array([d[key] for d in trafo_data_maps]), axis=0)
            for key in keys
        }
    elif clustering == "None":
        # Compute mean array for each key
        keys = trafo_data_maps[0].keys()
        avg_data = {
            key: np.nanmedian(np.array([d[key] for d in trafo_data_maps]), axis=0)
            for key in keys
        }
    else:
        raise ValueError(f'Clustering option {clustering} is not implemented!')

    # Creating the analysis dictionary
    structured_data = {
        'affine_matrices': affine_matrices,
        'template_contours': template_contours,  # Template contours but matched to respective measurement contours!
        'measurement_contours': measurement_contours,
        'measurement_trafo_contours': trafo_contours,
        'template_grid': template_grid,
        'measurement_grids': measurement_grids,
        'measurement_grids_shape': image_dims,
        "scale_matrices": scales,
        'verification_grids': verification_grids,
        'measurement_trafo_grids': trafo_grids,
        'verification_trafo_grids': trafo_ver_grids,
        'measurement_interpolated_grid': extended_grid,
        'measurement_interpolated_grid_shape': extended_grid_shape,
        'template_dataset': template_dataset,
        'measurement_datasets': measurement_datasets,
        'measurement_trafo_datasets': trafo_data_maps,
        'measurement_interpolated_dataset': avg_data,
        'measurement_filenames': filenames,
        'background_image': bg_images
    }

    return structured_data
