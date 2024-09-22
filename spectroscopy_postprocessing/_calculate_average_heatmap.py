import numpy as np


def average_heatmap(data_maps):
    arrays = [d['brillouin_shift_f_proj_trafo'] for d in data_maps]
    stacked_arrays = np.stack(arrays, axis=0)
    mean_array = np.median(stacked_arrays, axis=0)

    return mean_array
