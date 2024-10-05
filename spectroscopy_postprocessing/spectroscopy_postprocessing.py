import os
import matplotlib.pyplot as plt
import numpy as np
from _load_experiment import load_brillouin_experiment, load_afm_experiment
from _find_average_contour import find_average_contour
from _transform_2Dmap import transform_map2contour, transform_grid2contour
from _plot_maps import plot_maps_on_image, plot_contours, plot_cont_func, plot_average_heatmap
from _utilis import project_brillouin_dataset, export_analysis, import_analysis
from _br_afm_correlation import fit_coordinates_gmm, afm_brillouin_corr


def process_experiment(experiment, base_folder, results_folder, raw_data_key, load_experiment_func, label, cmap,
                       marker_size, vmin=None, vmax=None):
    os.makedirs(results_folder, exist_ok=True)

    folder_names, data_list, grid_list, grid_shape_list = [], [], [], []
    bf_data_list, mask_list = [], []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and ('#' in folder_name):
            folder_names.append(folder_name)

            # Load data from experiment folder
            data, grid, bf_data, mask = load_experiment_func(folder_path)

            if experiment == 'brillouin':
                # Create 2D Brillouin map from 3D dataset and ravel datasets
                data, grid = project_brillouin_dataset(data, grid)

            grid_shape_list.append(grid.shape)

            # Save imported data
            data_list.append(data)
            grid_list.append(grid)
            bf_data_list.append(bf_data)
            mask_list.append(mask)

    # Calculate average mask and plot result
    median_contour, contours_list, template_contour, matched_contour_list = find_average_contour(mask_list)
    fig = plot_contours(median_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform maps to average map
    data_trafo_list, grid_trafo_list, contour_trafo_list = [], [], []
    for index, contour in enumerate(contours_list):
        # Transform original grid to coordinate system of deformed contour
        trafo_grid_points, trafo_contour = transform_grid2contour(contour, median_contour, grid_list[index])

        # Transform maps to deformed coordinate system and interpolate on rectangular grid
        data_trafo = {}
        for key, data_map in data_list[index].items():
            data_map_trafo, extended_grid = transform_map2contour(trafo_grid_points, grid_list[0], data_map)
            data_trafo[key + '_trafo'] = data_map_trafo

        data_trafo_list.append(data_trafo)
        grid_trafo_list.append(trafo_grid_points)
        contour_trafo_list.append(trafo_contour)

        # Plot regular heatmaps
        fig = plot_maps_on_image(bf_data_list[index],
                                 data_list[index][f'{raw_data_key}'],
                                 grid_list[index],
                                 folder_names[index],
                                 label=label,
                                 cmap=cmap,
                                 marker_size=marker_size,
                                 vmin=vmin,
                                 vmax=vmax)
        output_path = os.path.join(results_folder, f'{folder_names[index]}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot transformed heatmap and contours
        fig = plot_cont_func(contour,
                             median_contour,
                             trafo_contour,
                             data_list[index][f'{raw_data_key}'],
                             data_trafo_list[index][f'{raw_data_key}_trafo'],
                             grid_list[index],
                             grid_trafo_list[index],
                             extended_grid,
                             label=label,
                             cmap=cmap,
                             marker_size=120,
                             vmin=vmin,
                             vmax=vmax)
        output_path = os.path.join(results_folder, f'MeanContourTransformed_{folder_names[index]}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Calculate average heatmap
    grid_avg, data_median_dict = fit_coordinates_gmm(grid_trafo_list, data_list)

    # Plot all transformed heatmaps and the averaged heatmap
    fig = plot_average_heatmap(data_list,
                               grid_trafo_list,
                               data_median_dict,
                               grid_avg,
                               median_contour,
                               f'{raw_data_key}',
                               label=label,
                               cmap=cmap,
                               marker_size=120,
                               vmin=vmin,
                               vmax=vmax)
    output_path = os.path.join(results_folder, f'Averaged_{raw_data_key.capitalize()}_Maps.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
            'brightfield_image': bf_data_list[i]
        }

    # Adding common data at folder level
    structured_data['median_contour'] = median_contour
    structured_data['template_contour'] = template_contour
    structured_data['average_data'] = data_median_dict
    structured_data['average_grid'] = grid_avg

    return structured_data


# Parameters for Brillouin data
brillouin_params = {
    "experiment": 'brillouin',
    "load_experiment_func": load_brillouin_experiment,
    "base_folder": 'C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Data',
    "results_folder": 'C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Analysis',
    "raw_data_key": 'brillouin_shift_f_proj',
    "label": 'Brillouin shift (GHz)',
    "cmap": 'viridis',
    "marker_size": 35,
    "vmin": None,
    "vmax": None
}

# Parameters for AFM data
afm_params = {
    "experiment": 'afm',
    "load_experiment_func": load_afm_experiment,
    "base_folder": 'C:/Users/niklas/OneDrive/Daten Master-Projekt/AFM/data/Xenopus_Brain/RawData',
    "results_folder": 'C:/Users/niklas/OneDrive/Daten Master-Projekt/AFM/data/Xenopus_Brain/results',
    "raw_data_key": 'modulus',
    "label": 'Reduced elastic modulus (Pa)',
    "cmap": 'hot',
    "marker_size": 5,
    "vmin": None,
    "vmax": None
}

# Process Brillouin data
brillouin_analysis_path = os.path.join(brillouin_params['results_folder'], 'BrillouinAnalysis.h5')
if not os.path.exists(brillouin_analysis_path):
    brillouin_analysis = process_experiment(**brillouin_params)
    export_analysis(brillouin_analysis_path, brillouin_analysis, brillouin_params)
else:
    brillouin_analysis, brillouin_params = import_analysis(brillouin_analysis_path)

# Process AFM data
afm_analysis_path = os.path.join(afm_params['results_folder'], 'AfmAnalysis.h5')
if not os.path.exists(afm_analysis_path):
    afm_analysis = process_experiment(**afm_params)
    export_analysis(afm_analysis_path, afm_analysis, afm_params)
else:
    afm_analysis, afm_params = import_analysis(afm_analysis_path)
# ToDo: CHECK if loading the dataset versus direct calculation yields different results!

# Extract relevant data for correlation map calculation
# ToDo: AFM and Brillouin grids in pixels and not µm!
afm_scale = 2
afm_analysis['average_grid'] = afm_analysis['average_grid'] * afm_scale
afm_analysis['median_contour'] = afm_analysis['median_contour'] * afm_scale

# Calculate correlation between AFM and Brillouin datasets
results_folder = './correlation'
os.makedirs(results_folder, exist_ok=True)
correlation, median_contour, afm_data_interp, br_data_interp, grid_avg = afm_brillouin_corr(afm_analysis, afm_params,
                                                                                            brillouin_analysis,
                                                                                            brillouin_params,
                                                                                            results_folder)
