# Parameters for AFM brain tissue stiffness maps
afm_params = {
    "overwrite_analysis": False,  # Whether to overwrite a pre-existing .h5 analysis file with the same name
    "contour_template": "average",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "outline_averaging": "mean",  # Methode used for outline averaging (if "contour_template": "average"): "star_domain", "mean", "median"
    "curvature": 1e-12,  # Curvature penalty used in DTW matching algorithm
    "bin_size": "None",  # Integer number used to bin regular data (e.g. images) to coarser blocks; can be "None"
    "contour_interp_n": 150,  # Number of points the contours will be interpolated to
    "clustering": "Median",  # "GMM": Use Gaussian Mixture Model to cluster data points and create average map; "Sum", "Median", "Mean": Use regular grid interpolation and average
    "afm_variables": ['modulus'],  # Choose AFM variables (column names in batchforce analysis file) to analyse
    "batchforce_filename": "data.csv",  # Filename of AFM batchforce analysis files
    "grid_conv_filename": "GridInversionMatrix.csv",  # File containing AFM grid conversion variables
    "boundary_filename": "brain_outline",  # Choose the .txt files containing the outline coordinates
    "key_point_filename": "None",  # Choose .txt file containing a common starting point on the outline; can be "None"
    "rot_axis_filename": "None"  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be "None"
}

# Parameters for synapse staining data
syn_params = {
    "overwrite_analysis": False,  # Whether to overwrite a pre-existing .h5 analysis file with the same name
    "contour_template": "average",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "outline_averaging": "mean",  # Methode used for outline averaging (if "contour_template": "average"): "star_domain", "mean", "median"
    "curvature": 1e-12,  # Curvature penalty used in DTW matching algorithm
    "bin_size": 15,  # Integer number used to bin regular data (e.g. images) to coarser blocks; can be "None"
    "contour_interp_n": 150,  # Number of points the contours will be interpolated to
    "clustering": "Median",  # "GMM": Use Gaussian Mixture Model to cluster data points and create average map; "Sum", "Median", "Mean": Use regular grid interpolation and average
    "syn_variables": ['Channel_2'],  # Choose Channel variables (Channel_X; with X numbered from 1 to N) to analyse
    "grid_conv_filename": "None",  # File containing AFM grid conversion variables
    "boundary_filename": "_BrainBoundary",  # Choose the .txt files containing the outline coordinates
    "key_point_filename": "None",  # Choose .txt file containing a common starting point on the outline; can be "None"
    "rot_axis_filename": "None"  # Choose .txt file containing two points along which the contours will be aligned (rotation); can be "None"
}

# Parameters for correlation analysis
corr_params = {
    "overwrite_analysis": False,  # Whether to overwrite a pre-existing .h5 analysis file with the same name
    "contour_template": "average",  # "first_element": Use first element in imported lists as template; "average": Calculate average contour and use as template
    "outline_averaging": "mean",  # Methode used for outline averaging (if "contour_template": "average"): "star_domain", "mean", "median"
    "curvature": 1e-12,  # Curvature penalty used in DTW matching algorithm
    "contour_interp_n": 150,  # Number of points the contours will be interpolated to
}
