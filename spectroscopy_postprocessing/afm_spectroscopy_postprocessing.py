import os
import matplotlib.pyplot as plt
import numpy as np
from _load_brillouin_experiment import load_afm_experiment, plot_afm_maps
from _find_average_contour import find_average_contour, plot_contours
from _transform_2Dmap import transform_map2contour_afm, plot_cont_func_afm
from _calculate_average_heatmap import average_heatmap, plot_average_heatmap

# Start analysis
base_folder = os.path.join('C:/Users/niklas/OneDrive/Daten Master-Projekt/AFM/data/Xenopus_Brain/RawData')
results_folder = os.path.join('./afm_results')
os.makedirs(results_folder, exist_ok=True)

data_list, data_grid_list, img_list, mask_list, experimental_folder = [], [], [], [], []
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ("#" in folder_name):
        experimental_folder.append(folder_name)

        data, img, mask = load_afm_experiment(folder_path)

        data_list.append(data['modulus'])
        data_grid_list.append(np.stack((data['x_image'], data['y_image']), axis=-1))
        img_list.append(img)
        mask_list.append(mask)

# Calculate average mask and plot result
median_contour, contours_list, template_contour, matched_contour_list = find_average_contour(mask_list)
fig = plot_contours(median_contour, template_contour, matched_contour_list)
output_path = os.path.join(results_folder, 'matched_mask_contours')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
# plt.close()

# Transform Brillouin maps to average map
bm_data_trafo_list, bm_grid_trafo_list, contour_trafo_list = [], [], []
for index, contour in enumerate(contours_list):
    # Plot regular heatmaps
    fig = plot_afm_maps(img_list[index],
                        data_list[index],
                        data_grid_list[index],
                        experimental_folder[index])
    output_path = os.path.join(results_folder, f'{experimental_folder[index]}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.show()
    # plt.close()

    data_trafo, trafo_grid, trafo_contour = transform_map2contour_afm(contour,
                                                                      median_contour,
                                                                      data_list[index],
                                                                      data_grid_list[index],
                                                                      data_grid_list[0])
    bm_data_trafo_list.append(data_trafo)
    bm_grid_trafo_list.append(trafo_grid)
    contour_trafo_list.append(trafo_contour)

    # Plot transformed heatmap and contours
    fig = plot_cont_func_afm(contour,
                             median_contour,
                             trafo_contour,
                             data_list[index],
                             data_grid_list[index],
                             bm_data_trafo_list[index],
                             bm_grid_trafo_list[index])

    output_path = os.path.join(results_folder, f'MeanContourTransformed_{experimental_folder[index]}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.show()
    # plt.close()

# Calculate average heatmap
bm_data_trafo_avg, bm_grid_trafo_avg = average_heatmap(bm_data_trafo_list, bm_grid_trafo_list)

# Plot all transformed heatmaps on top of each other into one plot and the averaged heatmap
fig = plot_average_heatmap(median_contour,
                           bm_data_trafo_list,
                           bm_data_trafo_avg,
                           data_grid_list,
                           bm_grid_trafo_avg)
output_path = os.path.join(results_folder, 'AveragedBrillouinMaps.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.show()
plt.close()
