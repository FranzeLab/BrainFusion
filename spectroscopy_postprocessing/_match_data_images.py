import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from _utilis import get_h5metadata, get_brillouin_data, get_user_input


def load_brillouin_experiment(folder_path):
    # Load Brillouin and image metadata
    h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    brillouin_metadata, br_rep_numbers = get_h5metadata(h5_path, 'Brillouin')

    # Load Brillouin data
    br_path = os.path.join(folder_path, 'Export')
    brillouin_data = get_brillouin_data(br_path)

    # Select Brillouin replicate if more than one available
    chosen_rep = 0
    if len(br_rep_numbers) == 1:
        chosen_rep = int(br_rep_numbers[0])
    else:
        # User input to select replicate
        user_input = None
        while user_input is None:
            user_input = get_user_input(f'Please enter a replicate number from: {br_rep_numbers} '
                                        f'(timeout in 10 seconds): ', timeout=10)
            if user_input and user_input.isdigit() and int(user_input) in br_rep_numbers:
                chosen_rep = int(user_input)
                break
            else:
                user_input = None
                print(f'Please provide a valid input!')

    # Extract chosen replicate
    brillouin_metadata_rep = brillouin_metadata[chosen_rep]
    brillouin_data_rep = brillouin_data[chosen_rep]

    return brillouin_data_rep, brillouin_metadata_rep


def load_brightfield_experiment(folder_path):
    img_path = os.path.join(folder_path, 'Plots')
    assert os.path.exists(img_path), f'{img_path} does not exist!'
    pattern = re.compile(r'Brillouin_FLrep(\d+)_channelBrightfield_aligned.png')

    img_dict = {}
    for filename in os.listdir(img_path):
        # Match the filename pattern
        match = pattern.match(filename)
        if match:
            rep = int(match.group(1))  # Extract the replicate number

            # Import the png file
            bg_path = os.path.join(img_path, filename)
            bg_img = Image.open(bg_path).convert('L')
            bg_img = np.rot90(np.array(bg_img), 3)

            # Initialize the replicate dictionary if it doesn't exist
            if rep not in img_dict:
                img_dict[rep] = {}

            # Assign the data slice to the appropriate slice number
            img_dict[rep] = bg_img

    # Load image metadata
    h5_path = os.path.join(folder_path, 'RawData', 'Brillouin.h5')
    assert os.path.exists(h5_path), f'{h5_path} does not exist!'
    img_metadata, img_rep_numbers = get_h5metadata(h5_path, 'Fluorescence')

    return img_dict[0], img_metadata[0]  # ToDo: Implement method to choose BF image


def get_mask(folder_path):
    mask_path = os.path.join(folder_path, 'Plots', 'brain_outline.png')
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)
        mask = np.rot90(mask, 3)
        return mask
    else:
        print(f'No mask was found for {folder_path}. Continuing without.')
        return None


def plot_results(background_img, data_extended, data_extended_grid, folder_name):
    # Create figure and axis
    fig, ax = plt.subplots()

    ax.imshow(background_img, origin='lower', cmap='gray', aspect='equal')

    # Plot heatmap
    heatmap = ax.scatter(data_extended_grid[:, :, 0],
                         data_extended_grid[:, :, 1],
                         c=data_extended,
                         cmap='viridis',
                         s=1,
                         label='Brillouin shift',
                         alpha=0.75,
                         vmax=5.45,
                         vmin=5.10
                         )
    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a colorbar and label it
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Brillouin shift (GHz)')

    return fig


def load_experiment(folder_path):
    assert os.path.exists(folder_path), f'{folder_path} does not exist!'

    # Load Brillouin data files and metadata
    brillouin_data_rep, brillouin_metadata_rep = load_brillouin_experiment(folder_path)

    # Load bright-field images and() metadata
    img_data_rep, img_metadata_rep = load_brightfield_experiment(folder_path)

    # Load mask
    mask = get_mask(folder_path)

    return brillouin_data_rep, brillouin_metadata_rep, img_data_rep, img_metadata_rep, mask
