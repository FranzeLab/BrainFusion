import os
import matplotlib.pyplot as plt
import numpy as np
from _find_average_contour import match_contours, calculate_median_contour
from _transform_2Dmap import transform_map2contour, transform_grid2contour
from _plot_maps import plot_contours, plot_cont_func, plot_corr_maps
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler


# Function to normalize heatmaps
def normalize_heatmap(heatmap):
    scaler = MinMaxScaler()
    return scaler.fit_transform(heatmap.reshape(-1, 1)).flatten()


def afm_brillouin_corr(afm_datasets, brillouin_datasets, afm_trafo_grids, brillouin_trafo_grids, afm_mean_contour,
                       brillouin_mean_contour, results_folder):
    # Make contour list
    contours_list = [afm_mean_contour, brillouin_mean_contour]

    # Circularly align and match contours using ellipse fitting
    matched_contour_list, template_contour = match_contours(contours_list)

    # Calculate the median contour
    median_contour = calculate_median_contour(matched_contour_list)

    fig = plot_contours(median_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_mask_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform maps to average map
    data_trafo_list, grid_trafo_list, contour_trafo_list = [], [], []
    for index, dataset in enumerate(afm_datasets):
        # Transform original grid to coordinate system of deformed contour
        afm_trafo_grid_points, trafo_contour = transform_grid2contour(contours_list[0], median_contour, afm_trafo_grids[index])

        grid_trafo_list.append(trafo_grid_points)
        contour_trafo_list.append(trafo_contour)




        # Plot transformed heatmap and contours
        fig = plot_cont_func(contour,
                             median_contour,
                             trafo_contour,
                             data_list[index],
                             data_trafo_list[index],
                             grid_list[index],
                             grid_trafo_list[index],
                             extended_grid,
                             label='Correlation maps',
                             cmap='YlGnBu',
                             marker_size=40,
                             vmin=None,
                             vmax=None)
        output_path = os.path.join(results_folder, f'MeanContourTransformed_{index}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Plot transformed heatmaps side by side
    fig = plot_corr_maps(median_contour,
                         brillouin_map=data_trafo_list[0],
                         afm_map=data_trafo_list[1],
                         grid=extended_grid,
                         mask=True,
                         marker_size=40,
                         vmin=None,
                         vmax=None)
    output_path = os.path.join(results_folder, f'MapsComparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr

    def bin_and_correlate(data_map1, data_map2, grid_points, bin_size=4):
        # Ensure both data maps are the same shape
        assert data_map1.shape == data_map2.shape, "Data maps must have the same shape."

        # Reshape the grid points and data maps for processing
        grid_points = grid_points.reshape(-1, 2)  # Assuming grid_points is Nx2
        data_map1 = data_map1.flatten()
        data_map2 = data_map2.flatten()

        # Create a KD-tree for nearest neighbor searching
        tree = cKDTree(grid_points)

        # Create bins by finding the closest points (4 points to one median)
        indices = tree.query(grid_points, k=bin_size)[1]  # Find the indices of 4 closest points

        # Initialize lists to hold median values
        median_map1 = []
        median_map2 = []

        # Calculate the median for each bin
        for i in range(len(grid_points)):
            if i % bin_size == 0:  # Take the first point of each bin
                # Extract the indices for the current bin
                current_indices = indices[i:i + bin_size]

                # Calculate the median values for both data maps
                median1 = np.nanmean(data_map1[current_indices])
                median2 = np.nanmean(data_map2[current_indices])

                median_map1.append(median1)
                median_map2.append(median2)

        # Convert lists to arrays
        median_map1 = np.array(median_map1)
        median_map2 = np.array(median_map2)

        mask = ~np.isnan(median_map1) & ~np.isnan(median_map2)

        # Filter out the NaNs using the mask
        median_map1 = median_map1[mask]
        median_map2 = median_map2[mask]

        # Calculate the Pearson correlation coefficient
        if len(data_trafo_list[0]) > 1 and len(data_trafo_list[1]) > 1:
            correlation, _ = pearsonr(median_map1, median_map2)
            print(f"Correlation: {correlation}")
        else:
            print("Not enough valid data points for correlation.")

        return correlation

    correlation_result = bin_and_correlate(data_trafo_list[0], data_trafo_list[1], extended_grid)
    print("Correlation after binning and taking medians:", correlation_result)

    # Mask map for Nans
    # Create a mask where neither heatmap has NaNs
    mask = ~np.isnan(data_trafo_list[0]) & ~np.isnan(data_trafo_list[1])

    # Filter out the NaNs using the mask
    data_trafo_list[0] = data_trafo_list[0][mask]
    data_trafo_list[1] = data_trafo_list[1][mask]

    # Calculate the Pearson correlation coefficient
    if len(data_trafo_list[0]) > 1 and len(data_trafo_list[1]) > 1:
        correlation, _ = pearsonr(data_trafo_list[0], data_trafo_list[1])
        print(f"Correlation: {correlation}")
    else:
        print("Not enough valid data points for correlation.")

    # Normalize both heatmaps
    heatmap1_norm = normalize_heatmap(data_trafo_list[0])
    heatmap2_norm = normalize_heatmap(data_trafo_list[1])

    # Plot the filtered data as a scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(heatmap1_norm, heatmap2_norm, color='blue', label='Data points')

    # Add labels and a title
    plt.xlabel('Brillouin shift')
    plt.ylabel('Reduced elastic modulus')
    plt.title('Correlation plot')

    # Show the plot
    plt.grid(True)
    plt.legend()
    plt.show()

    return correlation, extended_grid, median_contour
