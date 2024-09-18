import numpy as np
import matplotlib.path as mpath
import os.path
import h5py
import threading
import re
import pandas as pd
from PIL import Image


def get_user_input(prompt, timeout=10):
    user_input = [None]  # To store user input

    def ask_input():
        user_input[0] = input(prompt)

    # Create and start a thread to ask for input
    input_thread = threading.Thread(target=ask_input)
    input_thread.start()

    # Wait for the input thread to finish or timeout
    input_thread.join(timeout)

    # Return the user input if it was provided, otherwise return None
    return user_input[0]


def mask_contour(contour, grid):
    # Create the path from the contour
    path = mpath.Path(contour)

    # Find which points are inside the contour
    points = np.vstack((grid[:, :, 0].ravel(), grid[:, :, 1].ravel())).T
    mask_points = path.contains_points(points)

    # Reshape the result back to the shape of the mask
    mask = mask_points.reshape(grid[:, :, 0].shape)

    return mask


def center_contour(contour):
    # Calculate the centroid (center of mass) of the contour
    centroid = np.mean(contour, axis=0)

    # Translate all points so the centroid is at the origin
    centered_contour = contour - centroid

    return centered_contour, centroid


def create_data_grid(data):
    # Generate a grid covering the area for both contours
    x = np.arange(0, len(data[0, :]), 1)
    y = np.arange(0, len(data[:, 0]), 1)

    X, Y = np.meshgrid(x, y)  # Create grid of coordinates
    data_grid = np.stack((X, Y), axis=-1)

    return data_grid


def get_h5rep(h5_path, data_var):
    assert os.path.exists(h5_path), print(f'{h5_path} does not exist.')
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    with h5py.File(h5_path, 'r') as h5_file:
        brillouin_group = h5_file[data_var]
        rep_numbers = [int(key) for key in brillouin_group.keys() if key.isdigit()]

    return rep_numbers


def get_h5metadata(h5_path, data_var):
    assert os.path.exists(h5_path), f'{h5_path} does not exist.'
    assert data_var in ['Brillouin', 'Fluorescence'], (f'{data_var} is not available in h5 file. '
                                                       f'Please choose Brillouin or Fluorescence.')

    # Get number of replicates
    reps = get_h5rep(h5_path, data_var)
    assert isinstance(reps, list) and all(isinstance(rep, int) for rep in reps), (f'Reps {reps} '
                                                                                  f'must be a list of integers.')

    metadata_dict = {}
    with h5py.File(h5_path, 'r') as h5_file:
        for rep in reps:
            if data_var == 'Brillouin':
                # Get grid coordinates from Brillouin measurement (x,y inverted)
                brillouin_grid_x = h5_file[f'Brillouin/{rep}/payload/positions-x'][:]
                brillouin_grid_y = h5_file[f'Brillouin/{rep}/payload/positions-y'][:]
                brillouin_grid_z = h5_file[f'Brillouin/{rep}/payload/positions-z'][:]
                brillouin_grid = np.stack((brillouin_grid_x, brillouin_grid_y, brillouin_grid_z), axis=-1)
                brillouin_grid = np.transpose(brillouin_grid, (2, 1, 0, 3))
            else:
                brillouin_grid = None

            pixToMicrometerX = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixX']
            pixToMicrometerX_x, pixToMicrometerX_y = pixToMicrometerX.attrs['x'], pixToMicrometerX.attrs['y']

            pixToMicrometerY = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/micrometerToPixY']
            pixToMicrometerY_x, pixToMicrometerY_y = pixToMicrometerY.attrs['x'], pixToMicrometerY.attrs['y']

            origin = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/origin']
            origin_x, origin_y = origin.attrs['x'], origin.attrs['y']

            scanner = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionScanner']
            scanner_x, scanner_y = scanner.attrs['x'], scanner.attrs['y']

            stage = h5_file[f'{data_var}/{rep}/payload/scaleCalibration/positionStage']
            stage_x, stage_y = stage.attrs['x'], stage.attrs['y']

            # Create a dictionary for the current replicate


            replicate_metadata = {
                'pixToMicrometerX': np.stack((pixToMicrometerX_x, pixToMicrometerX_y), axis=-1),
                'pixToMicrometerY': np.stack((pixToMicrometerY_x, pixToMicrometerY_y), axis=-1),
                'origin': np.stack((origin_x, origin_y), axis=-1),
                'scanner': np.stack((scanner_x, scanner_y), axis=-1),
                'stage': np.stack((stage_x, stage_y), axis=-1),
                'brillouin_grid': brillouin_grid  # Switch positions
            }

            metadata_dict[rep] = replicate_metadata

    return metadata_dict, reps


def get_brillouin_data(br_path):
    assert os.path.exists(br_path), f'{br_path} does not exist!'
    pattern = re.compile(r'Brillouin_BMrep(\d+)_(\w+)_slice-(\d+)\.csv')

    data_dict = {}
    for filename in os.listdir(br_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number
            variable = match.group(2)  # Extract the variable name
            slice_num = int(match.group(3))  # Extract the slice number

            # Read the CSV file
            file_path = os.path.join(br_path, filename)
            data_slice = pd.read_csv(file_path, skiprows=2, header=None).to_numpy()

            # Expand the data_slice to add a new dimension for slices
            data_slice_expanded = np.expand_dims(data_slice, axis=-1)  # Shape becomes (rows, cols, 1)

            # Initialize the variable dictionary if it doesn't exist
            if rep not in data_dict:
                data_dict[rep] = {}

            # Initialize the variable array if it doesn't exist
            if variable not in data_dict[rep]:
                data_dict[rep][variable] = []

            # Append the expanded data slice to the appropriate variable for the replicate
            data_dict[rep][variable].append(data_slice_expanded)

    # Convert lists to arrays along the last dimension for each variable
    for rep in data_dict:
        for variable in data_dict[rep]:
            data_dict[rep][variable] = np.concatenate(data_dict[rep][variable], axis=-1)

    return data_dict

