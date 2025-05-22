import os
import re
import tifffile as tiff
from tifffile import imread, TiffFile
import numpy as np

from brainfusion._io import get_roi_from_txt
from brainfusion._utils import bin_single_image_channel

def load_hcr_experiment(folder_path, hcr_variables, data_filename='data.csv', key_point_filename="None",
                        rot_axis_filename="None", grid_conv_filename='GridInversionMatrix.csv',
                        boundary_filename='brain_outline', sampling_size="None", **kwargs):
    """
    Load maximum projected .tif files including all channels and tissue outlines.
    """
    # Load all image files
    tif_i_filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]

    # Import tif images
    tif_grids, tif_datasets, filenames = [], [], []
    for filename in tif_i_filenames:
        file_path = os.path.join(folder_path, filename)
        image = tiff.imread(file_path)

        # Get image dimensions
        height, width = image.shape[-2:]  # Always get last two dimensions as (H, W)

        # Generate pixel coordinates grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

        # Handle single-channel and multi-channel images
        channels = {}
        if len(image.shape) == 2:  # Single-channel
            channels['Channel_1'] = image.ravel()
        else:  # Multi-channel
            for i in range(image.shape[0]):
                channels[f'Channel_{i + 1}'] = image[i].ravel()

        # Randomly sample datasets for faster calculation
        if type(sampling_size) is int:
            print('Attention: Data sampling is activated to improve calculation time. Deactivate for proper analysis!')
            sample_idx = np.random.choice(len(pixel_grid), size=sampling_size, replace=False)
            pixel_grid = np.stack((pixel_grid[:, 0][sample_idx], pixel_grid[:, 1][sample_idx]), axis=1)
            channels = {key: c[sample_idx] for key, c in channels.items()}

        # Store data as dictionary
        tif_grids.append(pixel_grid)
        tif_datasets.append(channels)
        filenames.append(os.path.splitext(filename)[0])

    # Load all contour filenames corresponding to tif images
    tif_c_filenames = [f for f in os.listdir(folder_path) if f.endswith(boundary_filename + ".txt")]

    # Import contours corresponding to tif images
    tif_contours = []
    for index, filename in enumerate(tif_c_filenames):
        file_path = os.path.join(folder_path, filename)
        tif_contour = get_roi_from_txt(file_path, delimiter='\t', skip=1)
        tif_contours.append(tif_contour)

    # To make the boundary matching algorithm more robust, additional information like a landmark point similar on all
    # contours and an axis used to align contours can be included

    # Load all key-points in a list
    tif_keypoints = []
    keypoint_idx = 0  # If more than one keypoint is defined, use the first  # ToDo: Allow for multiple key-points
    if key_point_filename != "None":
        tif_p_filenames = [p for p in os.listdir(folder_path) if p.endswith(key_point_filename + ".txt")]
        for index, filename in enumerate(tif_p_filenames):
            file_path = os.path.join(folder_path, filename)
            tif_keypoint = get_roi_from_txt(file_path, delimiter='\t')[keypoint_idx]
            tif_keypoints.append(tif_keypoint)
    else:
        tif_keypoints = [None] * len(tif_contours)

    # Load all rotation axes in a list
    tif_axes = []
    if rot_axis_filename != "None":
        tif_r_filenames = [p for p in os.listdir(folder_path) if p.endswith(rot_axis_filename + ".txt")]
        for index, filename in enumerate(tif_r_filenames):
            file_path = os.path.join(folder_path, filename)
            tif_axis = get_roi_from_txt(file_path, delimiter='\t')
            tif_axes.append(tif_axis)
    else:
        tif_axes = [None] * len(tif_contours)

    scales = "None"
    bg_image = "None"

    results = {"grids": tif_grids,
               "scales": scales,
               "datasets": tif_datasets,
               "contours": tif_contours,
               "points": tif_keypoints,
               "axes": tif_axes,
               "filenames": filenames,
               "bg_images": bg_image}

    return results


def load_synapse_experiment(folder_path, syn_variables, key_point_filename="None", rot_axis_filename="None",
                            grid_conv_filename='GridInversionMatrix.csv', boundary_filename='BrainBoundary',
                            bin_size="None", bit_depth=16, normalize_percentile="None", clip=False, **kwargs):
    """
    Load .tif files including all channels and tissue outlines.
    """
    # Load all image files with corresponding brain outlines
    tif_i_filenames = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]

    # Import tif images
    tif_grids, tif_grids_dims, tif_datasets, tif_contours, filenames = [], [], [], [], []
    for filename in tif_i_filenames:
        file_path = os.path.join(folder_path, filename)

        # Load image data
        image = imread(file_path)

        # Load resolution metadata
        with TiffFile(file_path) as tif:
            page = tif.pages[0]
            x_res = page.coords['width'][1]
            y_res = page.coords['height'][1]

        # Get image dimensions
        height, width = image.shape[-2:]  # Always get last two dimensions

        # Generate pixel coordinates grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        pixel_grid = np.column_stack((xx.ravel(), yy.ravel()))

        # Handle single-channel and multi-channel images
        channels = {}
        if image.ndim == 2:  # Single-channel
            channels['Channel_1'] = image.ravel()
        else:  # Multi-channel
            for i in range(image.shape[0]):
                channels[f'Channel_{i + 1}'] = image[i].ravel()

        # Bin datasets for faster calculation
        if type(bin_size) is int:
            x_res, y_res = bin_size * x_res, bin_size * y_res
            print('Attention: Data binning is activated to improve calculation time!')
            for key, c in channels.items():
                binned_values, pixel_grid_tmp = bin_single_image_channel(c, pixel_grid, bin_size=bin_size)
                channels[key] = binned_values
            pixel_grid = pixel_grid_tmp

        # Renormalize image and clip high intensity values
        if isinstance(normalize_percentile, int):
            if bit_depth not in [8, 12, 16]:
                raise ValueError("bit_depth must be 8, 12, or 16")
            print(f'Attention: Channels are dynamically re-normalized to the {normalize_percentile}th percentile!')

            if clip is True:
                print(f'Attention: Intensity values above the {normalize_percentile}th percentile are clipped to max value!')

            max_val = 2 ** bit_depth - 1
            dtype = np.uint16 if bit_depth > 8 else np.uint8

            for key, c in channels.items():
                # Compute scaling value from percentile
                scale = np.percentile(c, normalize_percentile)
                if scale <= 0:
                    scaled = np.zeros_like(c, dtype=dtype)
                else:
                    scaled = c / scale * max_val

                # Optional clipping
                if clip is True:
                    scaled = np.clip(scaled, 0, max_val)

                # Convert to final integer type and store back
                channels[key] = scaled.astype(dtype)

        # Import contours corresponding to tif images
        contour_filename = filename.removesuffix(".tif") + boundary_filename + '.txt'
        contour_path = os.path.join(folder_path, contour_filename)
        tif_contour = get_roi_from_txt(contour_path, delimiter='\t', skip=1)

        # Store data as dictionary and contour
        tif_contours.append(tif_contour)
        tif_grids.append(pixel_grid)
        tif_grids_dims.append([height, width])
        tif_datasets.append(channels)
        filenames.append(os.path.splitext(filename)[0])

    # To make the boundary matching algorithm more robust, additional information like a landmark point similar on all
    # contours and an axis used to align contours can be included

    # Load all key-points in a list
    tif_keypoints = []
    keypoint_idx = 0  # If more than one keypoint is defined, use the first  # ToDo: Allow for multiple key-points
    if key_point_filename != "None":
        tif_p_filenames = [p for p in os.listdir(folder_path) if p.endswith(key_point_filename + ".txt")]
        for index, filename in enumerate(tif_p_filenames):
            file_path = os.path.join(folder_path, filename)
            tif_keypoint = get_roi_from_txt(file_path, delimiter='\t')[keypoint_idx]
            tif_keypoints.append(tif_keypoint)
    else:
        tif_keypoints = [None] * len(tif_contours)

    # Load all rotation axes in a list
    tif_axes = []
    if rot_axis_filename != "None":
        tif_r_filenames = [p for p in os.listdir(folder_path) if p.endswith(rot_axis_filename + ".txt")]
        for index, filename in enumerate(tif_r_filenames):
            file_path = os.path.join(folder_path, filename)
            tif_axis = get_roi_from_txt(file_path, delimiter='\t')
            tif_axes.append(tif_axis)
    else:
        tif_axes = [None] * len(tif_contours)

    scales = "None"
    bg_image = "None"

    results = {"grids": tif_grids,
               "scales": tif_grids_dims,
               "datasets": tif_datasets,
               "contours": tif_contours,
               "points": tif_keypoints,
               "axes": tif_axes,
               "filenames": filenames,
               "bg_images": bg_image}

    return results
