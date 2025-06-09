import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from brainfusion._match_contours import align_contours
from brainfusion._transform_2Dmap import transform_grid2contour
from brainfusion._plot_maps import plot_contours, plot_corr_maps


def project_brillouin_dataset(bm_data, bm_metadata, br_intensity_threshold=15):
    # ToDo: Overhaul this function!
    bm_data_proj = {}
    if 'brillouin_peak_intensity' in bm_data and 'brillouin_shift_f' in bm_data:
        # Filter out invalid peaks
        mask_peak = bm_data['brillouin_peak_intensity'] > br_intensity_threshold
        # Filter out water shifts
        mask_shift = (4.4 < bm_data['brillouin_shift_f']) & (bm_data['brillouin_shift_f'] < 10.0)

        # Check data distribution visually
        """
        import matplotlib.pyplot as plt
        cumulative_percentage = np.linspace(0, 100, len(sorted_data))
        plt.plot(sorted_data, cumulative_percentage, color='blue', linewidth=2)
        plt.title("Cumulative distribution")
        plt.xlabel("Brillouin shift (GHz)")
        plt.ylabel("Cumulative Percentage (%)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
        """

        mask = mask_peak & mask_shift & (0 < bm_data['brillouin_peak_fwhm_f']) & (bm_data['brillouin_peak_fwhm_f'] < 5)
    else:
        mask = True
    for key, value in bm_data.items():
        new_value = value.copy()  # Copy the original data to avoid modifying it

        # For distribution analysis
        sorted_data = np.sort(new_value.flatten())
        bm_data_proj[key + '_distribution'] = sorted_data  # Store the sorted distribution

        if key == 'brillouin_peak_intensity':
            continue

        new_value = np.where(mask, new_value, np.nan)

        proj_value = np.nanmedian(new_value, axis=-1).ravel()

        bm_data_proj[key + '_proj'] = proj_value  # Store the projection

    bm_grid_proj = bm_metadata['brillouin_grid'][:, :, 0, :2]  # Use x,y grid of first z-slice
    bm_grid_proj = np.column_stack([bm_grid_proj[:, :, 0].ravel(), bm_grid_proj[:, :, 1].ravel()])

    return bm_data_proj, bm_grid_proj



def bin_and_correlate(data_map1, data_map2, grid_points, bin_size=4, map1_fit_limits=None, map2_fit_limits=None):
    # Ensure both data maps are the same shape
    assert data_map1.shape == data_map2.shape, "Data maps must have the same shape."

    # Reshape the grid points and data maps for processing
    grid_points = grid_points.reshape(-1, 2)  # Assuming grid_points is Nx2
    data_map1 = data_map1.flatten()
    data_map2 = data_map2.flatten()

    # Create a KD-tree for nearest neighbor searching
    tree = cKDTree(grid_points)

    # Create bins by finding the closest points (4 points to one median)
    indices = tree.query(grid_points, k=bin_size)[1]  # Find the indices of closest points

    # Initialize lists to hold median values
    median_map1 = []
    median_map2 = []

    # Calculate the median for each bin
    for i in range(len(grid_points)):
        if i % bin_size == 0:  # Take the first point of each bin
            # Extract the indices for the current bin
            current_indices = indices[i:i + bin_size]

            # Calculate the median values for both data maps
            median1 = np.nanmedian(data_map1[current_indices])
            median2 = np.nanmedian(data_map2[current_indices])

            median_map1.append(median1)
            median_map2.append(median2)

    # Convert lists to arrays
    median_map1 = np.array(median_map1)
    median_map2 = np.array(median_map2)

    mask = ~np.isnan(median_map1) & ~np.isnan(median_map2)

    # Filter out the NaNs using the mask
    median_map1 = median_map1[mask]
    median_map2 = median_map2[mask]

    # Normalize both heatmaps
    heatmap1_norm = normalize_heatmap(median_map1)
    heatmap2_norm = normalize_heatmap(median_map2)

    # Filter boundaries
    mask1 = mask2 = np.full(heatmap1_norm.shape, True, dtype=bool)
    if map1_fit_limits:
        mask1 = (heatmap1_norm > map1_fit_limits[0]) & (heatmap1_norm < map1_fit_limits[1])

    if map2_fit_limits:
        mask2 = (heatmap2_norm > map2_fit_limits[0]) & (heatmap2_norm < map2_fit_limits[1])

    # Apply the combined mask
    combined_mask = mask1 & mask2
    heatmap1_norm_f, heatmap2_norm_f = heatmap1_norm[combined_mask], heatmap2_norm[combined_mask]

    # Calculate the Pearson correlation coefficient
    if len(heatmap1_norm_f) > 1 and len(heatmap2_norm_f) > 1:
        correlation, p_value = pearsonr(heatmap1_norm_f, heatmap2_norm_f)
        print("Correlation after binning and taking medians:", correlation)
    else:
        print("Not enough valid data points for correlation.")
        correlation = None

    return {'map1': heatmap1_norm, 'map2': heatmap2_norm, 'pearson': correlation, 'p_value': p_value}


def normalize_heatmap(heatmap):
    scaler = MinMaxScaler()
    return scaler.fit_transform(heatmap.reshape(-1, 1)).flatten()


def afm_brillouin_transformation(afm_analysis, afm_params, br_analysis, br_params, results_folder):
    # Extract data variables
    afm_data = afm_analysis['average_data']
    afm_grid = afm_analysis['average_grid']
    afm_contour = afm_analysis['average_contour']
    afm_key = f'{afm_params['raw_data_key']}_median'

    br_data = br_analysis['average_data']
    br_grid = br_analysis['average_grid']
    br_contour = br_analysis['average_contour']
    br_key = f'{br_params['raw_data_key']}_median'

    # Centre coordinate grids around origin
    afm_centre = np.mean(afm_grid, axis=0)
    afm_grid = afm_grid - afm_centre
    afm_contour = afm_contour - afm_centre

    br_centre = np.mean(br_grid, axis=0)
    br_grid = br_grid - br_centre
    br_contour = br_contour - br_centre

    # Create contours list
    contours_list = [afm_contour, br_contour]

    # Calculate average contour
    avg_contour, contours_list, template_contour, matched_contour_list, error_list = align_contours(
        contours_list=contours_list)

    # Plot the average contour
    fig = plot_contours(avg_contour, template_contour, matched_contour_list)
    output_path = os.path.join(results_folder, 'matched_average_contours.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Transform original AFM grid to coordinate system of average contour
    afm_trafo_grid, afm_trafo_contour = transform_grid2contour(contours_list[0],
                                                               avg_contour,
                                                               afm_grid)

    # Transform original Brillouin grid to coordinate system of average contour
    br_trafo_grid, br_trafo_contour = transform_grid2contour(contours_list[1],
                                                             avg_contour,
                                                             br_grid)

    # Find new coordinates to compare both grids
    trafo_grid_list = [afm_trafo_grid, br_trafo_grid]
    data_list = [afm_data, br_data]
    grid_avg, _, _ = fit_coordinates_gmm(trafo_grid_list, data_list, same_maps=False, num_components='min')

    # Transform data to new grid
    afm_data_interp, br_data_interp = {}, {}
    for key, value in afm_data.items():
        afm_data_interp[f'{key}'] = griddata(afm_trafo_grid, value, grid_avg, method='linear')
    for key, value in br_data.items():
        br_data_interp[f'{key}'] = griddata(br_trafo_grid, value, grid_avg, method='linear')

    # Plot transformed heatmaps side by side
    fig = plot_corr_maps(avg_contour,
                         afm_map=afm_data_interp[afm_key],
                         brillouin_map=br_data_interp[br_key],
                         grid=grid_avg,
                         mask=True,
                         marker_size=150,
                         vmin=None,
                         vmax=None)
    output_path = os.path.join(results_folder, f'MapComparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return afm_data_interp, br_data_interp, grid_avg, avg_contour
