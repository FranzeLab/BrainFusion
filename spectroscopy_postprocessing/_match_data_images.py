import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from skimage.transform import resize


def load_data(folder_path):
    data = []
    z_slices = []
    # Traverse the folder and look for .csv files
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                z_slice = int(file.split("slice-")[-1].split(".csv")[0])
                z_slices.append(z_slice)

                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path, skiprows=2, header=None)
                data.append(df.to_numpy())

    return z_slices, data


def match_images(bg_path, temp_path):
    # Load and match images as before
    assert os.path.exists(bg_path), f'{bg_path} does not exist!'
    bg_img = Image.open(bg_path).convert('L')
    bg_img = np.rot90(np.array(bg_img), 3)

    assert os.path.exists(temp_path), f'{temp_path} does not exist!'
    temp_img = Image.open(temp_path).convert('L')
    temp_img = np.rot90(np.array(temp_img), 3)

    result = cv2.matchTemplate(bg_img, temp_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    return bg_img, temp_img, (max_loc[1], max_loc[0])


def prepare_data(z_slices, data, template_shape):
    # Sort by z_slices and convert data to numpy array
    z_slices, data = zip(*sorted(zip(z_slices, data)))
    data = np.array(data)
    # data[data < 4.4] = np.nan
    data = np.max(data, axis=0)

    # Resize data to scale of template BF image (no interpolation)
    data_resized = resize(data, template_shape, order=0, anti_aliasing=False)

    return data, data_resized


def apply_mask(mask_path, background_img_shape, data):
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')  # Load mask in grayscale
        mask = np.array(mask) / 255.0  # Normalize mask (1 inside ROI, 0 outside)
        mask_resized = resize(mask, background_img_shape, order=0, anti_aliasing=False)
        mask_resized = np.rot90(mask_resized, 3)
        data = np.where(mask_resized == 1, data, np.nan)
        return mask_resized, data
    else:
        return None, None, data


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


def process_experiments(folder_path):
    z_slices, data = load_data(folder_path)
    bg_path = os.path.join(folder_path, 'Plots/Brillouin_FLrep0_channelBrightfield_aligned.png')
    temp_path = os.path.join(folder_path, 'Plots/Brillouin_FLrep0_channelBrightfield_BMrep0.png')
    mask_path = os.path.join(folder_path, 'Plots/brain_outline.png')

    # Open and match images
    background_img, template_img, translation = match_images(bg_path, temp_path)

    # Prepare data to same size as background image
    data, data_resized = prepare_data(z_slices, data, template_img.shape)
    data_extended = np.full(background_img.shape, np.nan)
    x1 = background_img.shape[0] - template_img.shape[0] - translation[0]
    x2 = background_img.shape[0] - translation[0]
    y1 = translation[1]
    y2 = template_img.shape[1] + translation[1]
    data_extended[x1:x2, y1:y2] = data_resized

    # Load and apply mask
    mask, data_extended = apply_mask(mask_path, background_img.shape, data_extended)
    return data, data_extended, mask, background_img
