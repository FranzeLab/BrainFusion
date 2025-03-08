import os
from brainfusion import *
from params import afm_params

# This script was made to explore the BrainFusion Python package using AFM stiffness maps of Xenopus brain tissue and
# in situ hybridization chain reaction mRNA staining images

# Set the base folder containing all AFM experiments (folder indicated with a # followed by a unique number)
base_folder = r'C:\Users\niklas\Documents\GitHub\BrainFusion\brainfusion\examples\Example_3_SM_TissueStiffnessHCR\data\AFM_Data'
afm_analysis_path = os.path.join(base_folder, 'BrainFusion_AFM_Analysis.h5')

# Run the analysis in case the resulting h5 has not been created yet or the results should be overwritten
if not os.path.exists(afm_analysis_path) or afm_params["overwrite_analysis"]:
    loaded = load_batchforce_all(base_path=base_folder, **afm_params)
    afm_analysis = fuse_measurement_datasets(base_path=base_folder, **loaded, **afm_params)
    export_analysis(afm_analysis_path, afm_analysis, afm_params)
# Import the pre-computed analysis file for further post-processing steps
else:
    afm_analysis, afm_params_loaded = import_analysis(afm_analysis_path)

    # Check if the currently defined analysis parameters match with the imported analysis parameters
    check_parameters(afm_params, afm_params_loaded)

# Plot the transformation results
plot_brainfusion_results(afm_analysis, **afm_params,
                         results_folder=os.path.join(base_folder, 'results'), key_quant='modulus',
                         label='', cmap='grey', marker_size=35, vmin=None, vmax=None, verify_trafo=True)
