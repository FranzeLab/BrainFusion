import os
import matplotlib.pyplot as plt
import numpy as np
from _utilis import bin_data, create_data_grid
from _match_data_images import process_experiments, plot_results
from _find_average_contour import find_average_contour, plot_contours
from _transform_2Dmap import transform_map2contour, plot_cont_func
from _calculate_average_heatmap import average_heatmap, plot_average_heatmap


# Set Brillouin grid dimensions (x,y,z) and resolution
img_dim = [460, 250, 70]
vox_dim = [20, 20, 10]
bin_factor = 8

# Start analysis
base_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin')
results_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Analysis/results')
os.makedirs(results_folder, exist_ok=True)

data_list, data_extended_list, data_extended_grid_list, mask_list, background_img_list, experimental_folder = [], [], [], [], [], []
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and folder_name != "Analysis":
        experimental_folder.append(folder_name)

        # Read in images and data, transform map size and position on image, apply mask
        data, data_extended, mask, background_img = process_experiments(folder_path)

        # Save results
        data_list.append(data)

        data_extended_bin = bin_data(data_extended, bin_factor)
        data_extended_list.append(data_extended_bin)
        data_extended_grid_list.append(create_data_grid(data_extended_bin))

        mask_list.append(bin_data(mask, bin_factor))
        background_img_list.append(bin_data(background_img, bin_factor))

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
    fig = plot_results(background_img_list[index],
                       data_extended_list[index],
                       data_extended_grid_list[index],
                       experimental_folder[index])
    output_path = os.path.join(results_folder, f'{experimental_folder[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    #fig.show()
    plt.close()

    data_map_trafo, trafo_grid, trafo_matched_contour = transform_map2contour(matched_contour,
                                                                              median_contour,
                                                                              data_extended_list[index],
                                                                              data_extended_grid_list[index])
    data_map_trafo_list.append(data_map_trafo)
    trafo_grid_list.append(trafo_grid)
    trafo_matched_contour_list.append(trafo_matched_contour)

    # Plot transformed heatmap and contours
    fig = plot_cont_func(matched_contour,
                         median_contour,
                         trafo_matched_contour,
                         data_extended_list[index],
                         data_extended_grid_list[index],
                         data_map_trafo,
                         trafo_grid)

    output_path = os.path.join(results_folder, f'MeanContourTransformed_{experimental_folder[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    #fig.show()
    plt.close()

# Calculate average heatmap
average_data_map = average_heatmap(data_map_trafo_list)

# Plot all transformed heatmaps on top of each other into one plot and the averaged heatmap
fig = plot_average_heatmap(data_map_trafo_list,
                           average_data_map,
                           data_extended_grid_list,
                           data_extended_grid_list[0],
                           median_contour)
output_path = os.path.join(results_folder, 'AveragedBrillouinMaps')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
plt.close()
