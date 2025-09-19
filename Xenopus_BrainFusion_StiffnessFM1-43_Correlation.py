import os
import numpy as np
from scipy.stats import fisher_exact
from scipy.spatial import cKDTree
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
from scipy.stats import pearsonr
from brainfusion import *
from params import corr_params

# Import AFM data
afm_base_path = r'.\data_afm\Stage_37_38\results'
afm_analysis_path = os.path.join(afm_base_path, 'BrainFusion_AFM_Analysis.h5')
afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)

# Import FM1-43 data
syn_base_path = r'.\data_synapse_staining\SM_pooled_data_#1_#2_HAnnotation\Stage_37_38\results_8bit_norm100percentile'
syn_analysis_path = os.path.join(syn_base_path, 'BrainFusion_FM1-43_Analysis.h5')
syn_analysis, syn_params_loaded = import_analysis(syn_analysis_path)

grids = [afm_analysis['measurement_interpolated_grid'], syn_analysis['measurement_interpolated_grid']*1-6]
scales = [afm_analysis['measurement_interpolated_grid_shape'], syn_analysis['measurement_interpolated_grid_shape']]
datasets = [{'correlation': afm_analysis['measurement_interpolated_dataset']['modulus']},
            {'correlation': syn_analysis['measurement_interpolated_dataset']['Channel_2']}
            ]
contours = [afm_analysis['template_contours'][2], syn_analysis['template_contours'][2]*1e-6]
points = [None, None]
axes = [None, None]
filenames = ['AFM', 'FM1-43']

# Correlation path
correlation_base_path = r'.\correlation'
correlation_analysis_path = os.path.join(correlation_base_path, "AFM_FM1-43_Correlation_Analysis.h5")

# Run the analysis in case the resulting .h5 has not been created yet or the results should be overwritten
if not os.path.exists(correlation_analysis_path) or corr_params["overwrite_analysis"]:
    fused_afm_fm143 = brain_fusion_correlation(base_path='Correlation: AFM - FM1-43', grids=grids,
                                               image_dims=["None", [318, 279]], scales=scales, datasets=datasets,
                                               contours=contours, points=points, axes=axes, filenames=filenames,
                                               **corr_params)
    export_analysis(correlation_analysis_path, fused_afm_fm143, corr_params)
# Import the pre-computed analysis file for further post-processing steps
else:
    fused_afm_fm143, corr_params_loaded = import_analysis(correlation_analysis_path)
    # Check if the currently defined analysis parameters match with the imported analysis parameters
    check_parameters(corr_params, corr_params_loaded)

plot_brainfusion_results(fused_afm_fm143, results_folder=r'./correlation', key_quant='correlation', cbar_label='',
                         cmap='hot', marker_size=200, vmin=None, vmax=None, image_dataset=False, verify_trafo=False,
                         mask=True, plot_background=False, correlation=True)

# Correlate AFM and FM1-43 maps
# Define correlation parameters
contour = fused_afm_fm143['template_contours'][0]

sparse_data = fused_afm_fm143['measurement_datasets'][0]['correlation']
dense_data = fused_afm_fm143['measurement_datasets'][1]['correlation']

sparse_grid = fused_afm_fm143['measurement_trafo_grids'][0]
dense_grid = fused_afm_fm143['measurement_trafo_grids'][1]

sparse_perc = dense_perc = 60

# Export AFM & FM1-43 data to .csv
df_afm = pd.DataFrame({"x": sparse_grid[:, 0], "y": sparse_grid[:, 1], "modulus": sparse_data})
df_afm.to_csv(r".\correlation\Averaged_AFM_Data_beforeThresholding.csv", index=False)

df_fm = pd.DataFrame({"x": dense_grid[:, 0], "y": dense_grid[:, 1], "FM143_normalized_intensity": dense_data})
df_fm.to_csv(r".\correlation\Averaged_FM1-43_Data_beforeThresholding.csv", index=False)

# Calculate correlation
# Mask sparse points inside contour
sparse_mask = mask_contour(contour, sparse_grid)
sparse_grid_filtered = sparse_grid[sparse_mask]
sparse_data_filtered = sparse_data[sparse_mask]

# Mask dense points inside contour
dense_mask = mask_contour(contour, dense_grid)
dense_grid_filtered = dense_grid[dense_mask]
dense_data_filtered = dense_data[dense_mask]

avg_dense_values, radii, result, stats_results, sparse_present, dense_present = (
    correlate_dense_around_sparse(sparse_data_filtered, sparse_grid_filtered, sparse_perc, dense_data_filtered,
                                  dense_grid_filtered, dense_perc, radius='max', average_func=np.nanmean))

# Export thresholded AFM & FM1-43 data to .csv
afm_labels = np.where(sparse_present, "stiff", "soft")
df_afm_thres = pd.DataFrame({"x": sparse_grid_filtered[:, 0], "y": sparse_grid_filtered[:, 1], "modulus_binary": afm_labels})
df_afm_thres.to_csv(r".\correlation\Averaged_AFM_Data_afterThresholding.csv", index=False)

fm_labels = np.where(dense_present, "high", "low")
df_fm_thres = pd.DataFrame({"x": sparse_grid_filtered[:, 0], "y": sparse_grid_filtered[:, 1], "FM143_normalized_intensity_binary": fm_labels})
df_afm_thres.to_csv(r".\correlation\Averaged_FM1-43_Data_afterThresholding.csv", index=False)

plot_correlation_distribution()
plot_correlation_with_radii(sparse_grid_filtered, dense_grid_filtered, contour, radii,
                            results_folder=correlation_base_path)
plot_correlation_masks(sparse_grid_filtered, sparse_present, dense_present, contour,
                       results_folder=correlation_base_path)

print(result)
print(stats_results)
