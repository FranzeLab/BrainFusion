import os
import numpy as np
from PIL import Image
import re
import pandas as pd
from ._utils import get_h5metadata, get_user_input, rotate3Dgrid


def choose_rep_number(rep_numbers):
    # Select Brillouin replicate if more than one available
    if len(rep_numbers) == 1:
        chosen_rep = int(rep_numbers[0])
    else:
        try:
            user_input = get_user_input(f'Please enter a replicate number from: {rep_numbers} '
                                        f'(timeout in 10 seconds): ', timeout=10)
            if user_input and user_input.isdigit() and int(user_input) in rep_numbers:
                chosen_rep = int(user_input)
                print(f'Valid input received. Script will continue!')
            else:
                raise ValueError("Invalid input or no input provided.")
        except Exception:
            # If user fails to provide valid input, choose the max
            chosen_rep = np.max(rep_numbers)
            print(f'No valid input received. Defaulting to max replicate: {chosen_rep}')

    return chosen_rep


def get_brillouin_data(bm_path):
    assert os.path.exists(bm_path), f'{bm_path} does not exist!'
    pattern = re.compile(r'Brillouin_BMrep(\d+)_(\w+)_slice-(\d+)\.csv')

    bm_dict = {}
    for filename in os.listdir(bm_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number
            variable = match.group(2)  # Extract the variable name
            slice_num = int(match.group(3))  # Extract the slice number

            # Read the CSV file
            file_path = os.path.join(bm_path, filename)
            data_slice = pd.read_csv(file_path, skiprows=2, header=None).to_numpy()
            data_slice = np.expand_dims(data_slice, axis=-1)  # Shape becomes (rows, cols, 1)

            # Initialize the variable dictionary if it doesn't exist
            if rep not in bm_dict:
                bm_dict[rep] = {}

            # Initialize the variable array if it doesn't exist
            if variable not in bm_dict[rep]:
                bm_dict[rep][variable] = []

            # Append the expanded data slice to the appropriate variable for the replicate
            bm_dict[rep][variable].append(data_slice)

    # Convert lists to arrays along the last dimension for each variable
    for rep in bm_dict:
        for variable in bm_dict[rep]:
            bm_dict[rep][variable] = np.concatenate(bm_dict[rep][variable], axis=-1)

    return bm_dict


def get_brightfield_data(bf_path):
    pattern = re.compile(r'Brillouin_FLrep(\d+)_channelBrightfield_aligned.png')
    bf_dict = {}
    for filename in os.listdir(bf_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number

            # Import the png file
            bf_path_join = os.path.join(bf_path, filename)
            bf_img = Image.open(bf_path_join).convert('L')
            bf_img = np.array(bf_img)

            # Initialize the replicate dictionary if it doesn't exist
            if rep not in bf_dict:
                bf_dict[rep] = {}

            # Assign the data slice to the appropriate slice number
            bf_dict[rep] = bf_img

    return bf_dict


def get_mask(folder_path):
    mask_path = os.path.join(folder_path, 'Plots', 'brain_outline.png')
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)
        return mask
    else:
        print(f'No mask was found for {folder_path}. Continuing without.')
        return None


def get_roi(folder_path, simple=False):
    if os.path.exists(folder_path):
        if simple:
            path = os.path.join(folder_path, 'Plots', 'brain_outline.txt')

            # Read the content of the .txt file
            paths_contents = {}
            with open(path, 'r') as f:
                # Store the content in a list
                content = f.readlines()
                # Remove newline characters and split coordinates
                coordinates = [tuple(line.strip().split('\t')) for line in content]

                # Convert to Nx2 NumPy array
                return np.array(coordinates, dtype=float)

        # Iterate over all items in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Initialize a list to hold the contents of .txt files for this folder
                paths_contents = {}

                # Iterate over all files in the subdirectory
                for file in os.listdir(item_path):
                    if file.endswith('.txt'):
                        # Construct the full file path
                        file_path = os.path.join(item_path, file)

                        # Read the content of the .txt file
                        with open(file_path, 'r') as f:
                            # Store the content in a list
                            content = f.readlines()
                            # Remove newline characters and split coordinates
                            coordinates = [tuple(line.strip().split(', ')) for line in content]

                            # Convert to Nx2 NumPy array
                            paths_contents[file] = np.array(coordinates, dtype=float)

                # Add the contents to the dictionary with the folder name as the key
                folder_dict[item] = paths_contents

        return folder_dict

    else:
        print(f'No .txt file was found for {folder_path}. Continuing without!')
        return None


def load_brillouin_experiment(folder_path):
    # Load Brillouin metadata file
    bm_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bm_metadata, bm_rep_numbers = get_h5metadata(bm_h5_path, 'Brillouin')

    # Load Brillouin data files
    bm_path = os.path.join(folder_path, 'Export')
    bm_data = get_brillouin_data(bm_path)

    # Choose replicate number and apply to metadata and data dicts
    bm_chosen_rep = choose_rep_number(bm_rep_numbers)
    bm_metadata_rep = bm_metadata[bm_chosen_rep]
    bm_data_rep = bm_data[bm_chosen_rep]

    # Load bright-field metadata file
    bf_h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    bf_metadata, bf_rep_numbers = get_h5metadata(bf_h5_path, 'Fluorescence')

    # Load bright-field data files
    bf_path = os.path.join(folder_path, 'Plots')
    bf_data = get_brightfield_data(bf_path)

    # Choose replicate number and apply to metadata and data dicts
    bf_chosen_rep = 0
    bf_data_rep = bf_data[bf_chosen_rep]

    # Load mask
    contour = get_roi(folder_path, simple=True)

    # Rotate bright-field image and mask
    bf_data_rep = np.fliplr(np.rot90(bf_data_rep, 1))

    # Flip image up-down for imshow function with origin=lower
    bf_data_rep = np.flipud(bf_data_rep)

    # Rotate data grid
    grid = bm_metadata_rep['brillouin_grid']
    rot_grid = grid.copy()
    rot_grid[:, :, :, 0], rot_grid[:, :, :, 1] = grid[:, :, :, 1], grid[:, :, :, 0]
    bm_metadata_rep['brillouin_grid'] = rot_grid

    # Check for correct positioning
    """
    import matplotlib.pyplot as plt
    scale_x = bm_metadata_rep['pixPerMicrometerX'][0, 1]
    scale_y = bm_metadata_rep['pixPerMicrometerY'][0, 0]

    plt.imshow(bf_data_rep, cmap='gray', origin='lower')
    plt.scatter(bm_metadata_rep['brillouin_grid'][:, :, 0, 0] * np.abs(scale_x),
                bm_metadata_rep['brillouin_grid'][:, :, 0, 1] * np.abs(scale_y),
                c=bm_data_rep['rayleigh_peak_intensity'][:, :, 0],
                cmap='viridis', vmin=1000, vmax=8000, s=5)
    plt.show()
    """

    # Brillouin scale to µm/pix
    assert np.isclose(np.abs(bm_metadata_rep['pixPerMicrometerX'][0, 1]),
                      np.abs(bm_metadata_rep['pixPerMicrometerY'][0, 0]),
                      rtol=1.e-3)
    scale = 1 / np.abs(bm_metadata_rep['pixPerMicrometerX'][0, 1])

    return bm_data_rep, bm_metadata_rep, bf_data_rep, contour, scale


def load_afm_experiment(folder_path):
    # Load the AFM CSV file and extract data and grid coordinates
    data_path = os.path.join(folder_path, 'region analysis', 'data.csv')
    data = pd.read_csv(data_path)
    afm_data = {'modulus': data['modulus'], 'beta_pyforce': data['beta_pyforce'],
                'k0_pyforce': data['k0_pyforce'], 'k_pyforce': data['k_pyforce']}
    afm_data = {key: np.array(value) for key, value in afm_data.items()}

    # Calculate scaling factor in micrometer/pix
    scale = data['m_per_pix'][0] * 1e-6

    # Load AFM grid and scale to µm
    afm_grid = np.stack((np.array(data['x_image']), np.array(data['y_image'])), axis=-1)

    # Load background image
    img_path = os.path.join(folder_path, 'Pics', 'calibration', 'overview.tif')
    img = Image.open(img_path).convert('L')
    img = np.array(img)

    # Load mask
    mask_path = os.path.join(folder_path, 'Pics', 'calibration')
    orientation = []
    if os.path.exists(os.path.join(mask_path, 'brain_outline_OriLeft.png')):
        orientation.append('left')

        img = np.flipud(img)
        afm_grid[:, 1] = img.shape[0] - afm_grid[:, 1]

        mask_path = os.path.join(mask_path, 'brain_outline_OriLeft.png')
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)
        mask = np.flipud(mask)

    elif os.path.exists(os.path.join(mask_path, 'brain_outline_OriRight.png')):
        orientation.append('right')

        mask_path = os.path.join(mask_path, 'brain_outline_OriRight.png')
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)

    else:
        print(f'No matching mask was found for {folder_path}!')
        exit()

    # Scale AFM grid
    afm_grid = afm_grid * scale

    # Check for correct positioning
    """
    import matplotlib.pyplot as plt
    
    height_in_mu = img.shape[0] * scale
    width_in_mu = img.shape[1] * scale

    # Plot background image and heatmap
    plt.imshow(img, cmap='gray', aspect='equal', origin='lower', extent=[0, width_in_mu, 0, height_in_mu])
    plt.scatter(afm_grid[:, 0], afm_grid[:, 1], c=afm_data['modulus'],
                cmap='viridis', vmin=1000, vmax=8000, s=5)
    plt.show()
    """

    return afm_data, afm_grid, img, mask, scale
