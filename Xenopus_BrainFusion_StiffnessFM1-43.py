import os
from brainfusion import *
from params import afm_params, syn_params


#--- AFM ANALYSIS ---#
# Set the base folder containing all AFM experiments (folder indicated with a # followed by a unique number)
base_folder = r'.\data_afm'
results_folder_name = 'results_test'
stages = [f for f in os.listdir(base_folder) if 'Stage' in f]

for stage_folder in stages:
    afm_base_path = os.path.join(base_folder, stage_folder)
    afm_analysis_path = os.path.join(afm_base_path, results_folder_name, r'BrainFusion_AFM_Analysis.h5')

    # Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
    if not os.path.exists(afm_analysis_path) or afm_params["overwrite_analysis"]:
        loaded = load_batchforce_all(base_path=afm_base_path, **afm_params)
        afm_analysis = brain_fusion(**loaded, **afm_params)
        export_analysis(afm_analysis_path, afm_analysis, afm_params)
    # Import the pre-computed analysis file for further post-processing steps
    else:
        afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)
        # Check if the currently defined analysis parameters match with the imported analysis parameters
        check_parameters(afm_params, afm_params_loaded)

    # Plot the transformation results
    plot_brainfusion_results(afm_analysis, **afm_params,
                             results_folder=os.path.join(afm_base_path, results_folder_name), key_quant='modulus',
                             cbar_label='', cmap='hot', marker_size=300, vmin=0, vmax=400, image_dataset=False,
                             verify_trafo=False, mask=True, plot_background=True)

    # Export AFM & FM1-43 data to .csv
    df_afm = pd.DataFrame({"x": sparse_grid[:, 0], "y": sparse_grid[:, 1], "modulus": sparse_data})
    df_afm.to_csv(r".\correlation\Averaged_AFM_Data.csv", index=False)

#--- FM1-43 ANALYSIS ---#
# Set the base folder containing all synapse staining imaging data
base_folder = r'.\data_synapse_staining\SM_pooled_data_#1_#2_HAnnotation'
results_folder_name = 'results_8bit_norm100percentile_test'
stages = [f for f in os.listdir(base_folder) if 'Stage' in f]

for stage_folder in stages:
    syn_base_path = os.path.join(base_folder, stage_folder)
    syn_analysis_path = os.path.join(syn_base_path, results_folder_name, 'BrainFusion_FM1-43_Analysis.h5')

    # Run the analysis in case the resulting .h5 has not been created yet or the results should be overwritten
    if not os.path.exists(syn_analysis_path) or syn_params["overwrite_analysis"]:
        loaded = load_synapse_experiment(folder_path=syn_base_path, **syn_params, bit_depth=8, clip=False,
                                         normalize_percentile=100)
        syn_analysis = brain_fusion(**loaded, **syn_params)
        export_analysis(syn_analysis_path, syn_analysis, syn_params)
    # Import the pre-computed analysis file for further post-processing steps
    else:
        syn_analysis, syn_params_loaded = import_analysis(syn_analysis_path)
        # Check if the currently defined analysis parameters match with the imported analysis parameters
        check_parameters(syn_params, syn_params_loaded)

    # Plot the transformation results
    plot_brainfusion_results(syn_analysis, **syn_params,
                             results_folder=os.path.join(syn_base_path, results_folder_name), key_quant='Channel_2',
                             cbar_label='', cmap='Greys', marker_size=10, vmin=None, vmax=None, image_dataset=True,
                             verify_trafo=False, mask=True)
