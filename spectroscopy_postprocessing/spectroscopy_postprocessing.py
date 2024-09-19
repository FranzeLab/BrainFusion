import os
import matplotlib.pyplot as plt
from _load_brillouin_experiment import load_brillouin_experiment, plot_brillouin_maps
from _find_average_contour import find_average_contour, plot_contours
from _transform_2Dmap import transform_map2contour, plot_cont_func
from _calculate_average_heatmap import average_heatmap, plot_average_heatmap
from _utilis import project_brillouin_dataset

# Start analysis
base_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Data')
results_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/Brillouin/Analysis/results_new')
os.makedirs(results_folder, exist_ok=True)

folder_names, bf_data_list, bf_metadata_list, mask_list = [], [], [], []
bm_data_list, bm_metadata_list = [], []

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        folder_names.append(folder_name)

        # Load data from experiment folder including Brillouin maps and images
        bm_data, bm_metadata, bf_data, bf_metadata, mask = load_brillouin_experiment(folder_path)

        # Create 2D Brillouin map from 3D dataset and save in dict
        bm_data_proj = project_brillouin_dataset(bm_data)

        # Save results
        bm_data_list.append(bm_data_proj)
        bm_metadata_list.append(bm_metadata)
        bf_data_list.append(bf_data)
        bf_metadata_list.append(bf_metadata)
        mask_list.append(mask)

# Calculate average mask and plot result
median_contour, contours_list, template_contour, matched_contour_list = find_average_contour(mask_list)
fig = plot_contours(median_contour, template_contour, matched_contour_list)
output_path = os.path.join(results_folder, 'matched_mask_contours')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
#plt.close()

# Transform Brillouin maps to average map
bm_data_trafo_list, bm_grid_trafo_list, contour_trafo_list = [], [], []
for index, contour in enumerate(contours_list):
    # Plot regular heatmaps
    fig = plot_brillouin_maps(bf_data_list[index],
                              bm_data_list[index],
                              bm_metadata_list[index],
                              folder_names[index])
    output_path = os.path.join(results_folder, f'{folder_names[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.show()
    #plt.close()

    bm_data_trafo, trafo_grid, trafo_contour = transform_map2contour(contour,
                                                                     median_contour,
                                                                     bm_data_list[index],
                                                                     bm_metadata_list[index])
    bm_data_trafo_list.append(bm_data_trafo)
    bm_grid_trafo_list.append(trafo_grid)
    contour_trafo_list.append(trafo_contour)

    # Plot transformed heatmap and contours
    fig = plot_cont_func(contour,
                         median_contour,
                         trafo_contour,
                         bm_data_list[index],
                         bm_metadata_list[index],
                         bm_data_trafo_list[index],
                         bm_grid_trafo_list[index])

    output_path = os.path.join(results_folder, f'MeanContourTransformed_{folder_names[index]}')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.show()
    #plt.close()

# Calculate average heatmap
bm_data_trafo_list = average_heatmap(bm_data_trafo_list)

# Plot all transformed heatmaps on top of each other into one plot and the averaged heatmap
fig = plot_average_heatmap(data_map_trafo_list,
                           average_data_map,
                           bm_grid_list,
                           bm_grid_list[0],
                           median_contour)
output_path = os.path.join(results_folder, 'AveragedBrillouinMaps')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
plt.close()
