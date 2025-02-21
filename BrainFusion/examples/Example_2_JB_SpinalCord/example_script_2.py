from BrainFusion import *
from params import afm_params

# This script is made to explore the BrainFusion Python package using AFM stiffness maps of spinal cord tissue and image
# data quantifying myelin staining

base_folder = afm_params['base_folder']

for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path) and ('#' in folder_name):
        # Define where the results of this analysis are stored
        afm_analysis_path = os.path.join(folder_path, 'BrainFusion_AFM_Analysis.h5')

        # Run the analysis in case the resulting h5 has not been created yet
        if not os.path.exists(afm_analysis_path):
            afm_analysis = process_single_experiments(**afm_params, folder_name=folder_name)
            export_analysis(afm_analysis_path, afm_analysis, afm_params)
        # Import the pre-computed analysis file for further post-processing steps
        else:
            afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)
            check_parameters(afm_params, afm_params_loaded)

        # Plot results
        plot_sc_experiments(afm_analysis, **afm_params, results_folder=os.path.join(base_folder, folder_name, 'results'),
                            label='', cmap='grey', marker_size=20, vmin=None, vmax=None)
