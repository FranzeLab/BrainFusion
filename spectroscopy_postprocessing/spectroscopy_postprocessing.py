import os
import matplotlib.pyplot as plt
import numpy as np
from ._find_average_contour import find_average_contour, extract_and_interpolate_contours, interpolate_contour
from ._transform_2Dmap import transform_map2contour, transform_grid2contour
from ._plot_maps import plot_contours
from ._utils import project_brillouin_dataset
from ._br_afm_correlation import fit_coordinates_gmm


def process_experiment(experiment, base_folder, results_folder, load_experiment_func, **kwargs):
    os.makedirs(results_folder, exist_ok=True)

    folder_names, data_list, grid_list, grid_shape_list = [], [], [], []
    bf_data_list, mask_list, scale_list = [], [], []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and ('#' in folder_name):
            folder_names.append(folder_name)

            # Load data from experiment folder
            data, grid, bf_data, mask, pix_per_um = load_experiment_func(folder_path)

            if experiment == 'brillouin':
                # Create 2D Brillouin map from 3D dataset and ravel datasets
                data, grid = project_brillouin_dataset(data, grid)

            grid_shape_list.append(grid.shape)

            # Save imported data
            data_list.append(data)
            grid_list.append(grid)
            bf_data_list.append(bf_data)
            mask_list.append(mask)
            scale_list.append(pix_per_um)

    # Transform mask to contour and scale
    if isinstance(mask_list[0], np.ndarray) and mask_list[0].ndim == 2 and mask_list[0].shape[1] == 2:
        contours = mask_list
        contours_list = []
        for contour in contours:
            contour[:, 1] = 1023 - contour[:, 1]  # ToDo: FIX
            contours_list.append(interpolate_contour(contour, num_points=1000))
    else:
        contours_list = extract_and_interpolate_contours(mask_list, num_points=1000)
    contours_list = [contour * scale_list[index] for index, contour in enumerate(contours_list)]

    # Calculate average mask and plot result
    med_contour, contours_list, template_contour, matched_contour_list, error_list = find_average_contour(contours_list)

    fig = plot_contours(med_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform maps to average map
    data_trafo_list, grid_trafo_list, contour_trafo_list, extended_grid = [], [], [], []
    for index, contour in enumerate(contours_list):
        # Transform original grid to coordinate system of deformed contour
        trafo_grid_points, trafo_contour = transform_grid2contour(contour, med_contour, grid_list[index])

        # Transform maps to deformed coordinate system and interpolate on rectangular grid
        data_trafo = {}
        for key, data_map in data_list[index].items():
            data_map_trafo, extended_grid = transform_map2contour(trafo_grid_points, grid_list[0], data_map)
            data_trafo[key + '_trafo'] = data_map_trafo

        data_trafo_list.append(data_trafo)
        grid_trafo_list.append(trafo_grid_points)
        contour_trafo_list.append(trafo_contour)

    # Calculate average heatmap
    grid_avg, data_median_dict = fit_coordinates_gmm(grid_trafo_list, data_list, same_maps=True, num_components='mean')

    # Creating the analysis dictionary
    structured_data = {}
    for i, folder in enumerate(folder_names):
        structured_data[folder] = {
            'raw_data': data_list[i],  # For Brillouin data, this is already projected
            'trafo_data': data_trafo_list[i],
            'raw_grid': grid_list[i],  # For Brillouin data, this is already projected
            'grid_shape': grid_shape_list[i],
            'trafo_grid': grid_trafo_list[i],
            'contour': contours_list[i],
            'trafo_contour': contour_trafo_list[i],
            'brightfield_image': bf_data_list[i],
            'pix_per_um': scale_list[i]
        }

    # Adding common data at folder level
    structured_data['median_contour'] = med_contour
    structured_data['template_contour'] = template_contour
    structured_data['average_data'] = data_median_dict
    structured_data['average_grid'] = grid_avg
    structured_data['extended_grid'] = extended_grid

    return structured_data
