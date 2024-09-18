import os
import matplotlib.pyplot as plt
import numpy as np
from _utilis import create_data_grid
from _match_data_images import load_experiment, plot_results
from _find_average_contour import find_average_contour, plot_contours
from _transform_2Dmap import transform_map2contour, plot_cont_func
from _calculate_average_heatmap import average_heatmap, plot_average_heatmap


# Start analysis
base_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Data')
results_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Analysis/results_new')
os.makedirs(results_folder, exist_ok=True)

folder_names, data_list, metadata_list, img_data_list, img_metadata_list, mask_list, br_grid_list, br_shift_proj_list = [], [], [], [],  [], [], [], []
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        folder_names.append(folder_name)

        # Load data from experiment folder including Brillouin maps and images
        br_data, br_metadata, img_data, img_metadata, mask = load_experiment(folder_path)

        # Transform Brillouin grid coordinates to image coordinates
        brillouin_grid = br_metadata['brillouin_grid']
        stage_coord = br_metadata['stage']
        brillouin_grid[:, :, 0, 0] = brillouin_grid[:, :, 0, 0] - stage_coord[0, 0]  # x-grid
        brillouin_grid[:, :, 0, 1] = brillouin_grid[:, :, 0, 1] - stage_coord[0, 1]  # y-grid

        # Project data along z-axis
        # ToDo: Implement more elaborate analysis including z-dimension
        br_shift = br_data['brillouin_shift_f']
        br_shift[br_shift < 4.4] = np.nan
        br_shift_proj = np.max(br_shift, axis=-1)

        # Save results
        data_list.append(br_data)
        metadata_list.append(br_metadata)
        img_data_list.append(img_data)
        img_metadata_list.append(img_metadata)
        mask_list.append(mask)
        br_grid_list.append(brillouin_grid[:, :, 0, :])
        br_shift_proj_list.append(br_shift_proj)

# Calculate average mask and plot result
median_contour, matched_contour_list = find_average_contour(mask_list)
fig = plot_contours(matched_contour_list[0], matched_contour_list, median_contour)
output_path = os.path.join(results_folder, 'matched_mask_contours')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
#fig.show()
plt.close()

# Transform Brillouin maps to average map
data_map_trafo_list, trafo_grid_list, trafo_matched_contour_list = [], [], []
for index, matched_contour in enumerate(matched_contour_list):
    # Plot regular heatmaps
    fig = plot_results(img_data_list[index],
                       br_shift_proj_list[index],
                       br_grid_list[index][:, :, :],
                       folder_names[index])
    output_path = os.path.join(results_folder, f'{folder_names[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    #fig.show()
    plt.close()

    data_map_trafo, trafo_grid, trafo_matched_contour = transform_map2contour(matched_contour,
                                                                              median_contour,
                                                                              br_shift_proj_list[index],
                                                                              br_grid_list[index])
    data_map_trafo_list.append(data_map_trafo)
    trafo_grid_list.append(trafo_grid)
    trafo_matched_contour_list.append(trafo_matched_contour)

    # Plot transformed heatmap and contours
    fig = plot_cont_func(matched_contour,
                         median_contour,
                         trafo_matched_contour,
                         br_shift_proj_list[index],
                         br_grid_list[index],
                         data_map_trafo,
                         trafo_grid)

    output_path = os.path.join(results_folder, f'MeanContourTransformed_{folder_names[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    #fig.show()
    plt.close()

# Calculate average heatmap
average_data_map = average_heatmap(data_map_trafo_list)

# Plot all transformed heatmaps on top of each other into one plot and the averaged heatmap
fig = plot_average_heatmap(data_map_trafo_list,
                           average_data_map,
                           br_grid_list,
                           br_grid_list[0],
                           median_contour)
output_path = os.path.join(results_folder, 'AveragedBrillouinMaps')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
plt.close()
