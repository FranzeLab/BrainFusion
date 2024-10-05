import matplotlib.pyplot as plt
import numpy as np
from _utilis import mask_contour, scatter_with_touching_squares


def plot_contours(median_contour, template_contour, matched_contours):
    """Plot the template, matched contours, and the median contour"""
    fig = plt.figure(figsize=(8, 8))

    # Plot the template contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], 'g-', linewidth=2, label='Mask (Template)')
    plt.scatter(template_contour[0, 0], template_contour[0, 1], color='orange', s=12, label='First contour coordinate')

    # Plot the matched contours (only label the first one)
    for i, contour in enumerate(matched_contours[1:]):
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], 'b-', label='Masks (Matched)')
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)
        else:
            plt.plot(contour[:, 0], contour[:, 1], 'b-')  # No label for subsequent contours
            plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)

    # Plot the median contour
    plt.plot(median_contour[:, 0], median_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    plt.legend()
    plt.axis('equal')
    plt.title('Path Density with Contours')

    return fig


def plot_maps_on_image(img, data, grid, folder_name, label='Brillouin shift (GHz)', cmap='viridis', marker_size=15,
                       vmin=None, vmax=None):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot background image and heatmap
    ax.imshow(img, cmap='gray', aspect='equal', origin='lower')
    heatmap = ax.scatter(grid[:, 0],
                         grid[:, 1],
                         c=data,
                         cmap=cmap,
                         s=marker_size,
                         alpha=0.75,
                         vmin=vmin,
                         vmax=vmax
                         )

    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(label)

    return fig


def plot_cont_func(original_contour, deformed_contour, trafo_contour, bm_data, bm_data_trafo_list,
                   bm_grid_points, grid_points_trafo, extended_grid, label='Brillouin shift (GHz)', cmap='viridis',
                   marker_size=30, vmin=None, vmax=None):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original grid
    axes[0].scatter(bm_grid_points[:, 0],
                    bm_grid_points[:, 1],
                    c=bm_data,
                    cmap=cmap,
                    s=marker_size,
                    label='Original Grid',
                    vmin=vmin,
                    vmax=vmax)

    axes[0].legend()
    axes[0].set_title('Original data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using new coordinates and transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    heatmap_trafo_coords = axes[1].scatter(grid_points_trafo[:, 0],
                                           grid_points_trafo[:, 1],
                                           c=bm_data,
                                           cmap=cmap,
                                           s=marker_size,
                                           label='Transformed Grid (New coordinates)',
                                           alpha=1,
                                           vmin=vmin,
                                           vmax=vmax)
    heatmap_griddata = axes[1].scatter(extended_grid[:, 0],
                                       extended_grid[:, 1],
                                       c=bm_data_trafo_list,
                                       cmap=cmap,
                                       s=marker_size,
                                       label='Transformed Grid (Griddata)',
                                       alpha=0,
                                       vmin=vmin,
                                       vmax=vmax)

    axes[1].legend()
    axes[1].set_title('Transformed data map')
    axes[1].grid()
    axes[1].axis('equal')

    # Plot colorbar
    cbar = fig.colorbar(heatmap_trafo_coords, ax=axes[1])
    cbar.set_label(label)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_average_heatmap(data_maps, grid_trafos, data_avg, grid_avg, average_contour, data_variable,
                         label='Brillouin shift (GHz)', cmap='viridis', marker_size=15, vmin=None, vmax=None, mask=True):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Median contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Heatmaps
    for i, _ in enumerate(data_maps[1:]):
        if i == 0:
            axes[0].scatter(grid_trafos[i][:, 0],
                            grid_trafos[i][:, 1],
                            c=data_maps[i][data_variable],
                            cmap=cmap,
                            s=marker_size,
                            label='Transformed Grid',
                            alpha=0.5,
                            vmin=vmin,
                            vmax=vmax)
        else:
            axes[0].scatter(grid_trafos[i][:, 0],
                            grid_trafos[i][:, 1],
                            c=data_maps[i][data_variable],
                            cmap=cmap,
                            s=marker_size,
                            alpha=0.5,
                            vmin=vmin,
                            vmax=vmax)

    axes[0].legend()
    axes[0].set_title('Layered data maps of transformed spatial maps')
    axes[0].grid()
    axes[0].axis('equal')

    # Average heatmap
    # Median contour
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Mask average data
    assert type(mask) is bool, print('The provided mask argument is not boolean!')
    if mask is True:
        mask = mask_contour(average_contour, grid_avg)
    else:
        mask = np.full(grid_avg.shape[0], True)

    # Average data
    heatmap = axes[1].scatter(np.ma.masked_where(~mask, grid_avg[:, 0]),
                              np.ma.masked_where(~mask, grid_avg[:, 1]),
                              c=data_avg[f'{data_variable}_median'],
                              cmap=cmap,
                              s=marker_size,
                              marker='s',
                              label='Transformed Grid',
                              alpha=1,
                              vmin=vmin,
                              vmax=vmax)

    axes[1].legend()
    axes[1].set_title('Average data map of transformed spatial maps')
    axes[1].grid()
    axes[1].axis('equal')

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=axes[1])
    cbar.set_label(label)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_corr_maps(median_contour, afm_map, brillouin_map, grid, mask=True, marker_size=40, vmin=None, vmax=None):
    # Mask average data
    assert type(mask) is bool, print('The provided mask argument is not boolean!')
    if mask is True:
        mask = mask_contour(median_contour, grid)
    else:
        mask = np.full(grid.shape[0], True)

    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the AFM map
    axes[0].plot(median_contour[:, 0], median_contour[:, 1], 'r-', label='Average Contour')
    heatmap_brillouin = axes[0].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                        np.ma.masked_where(~mask, grid[:, 1]),
                                        c=afm_map,
                                        cmap='hot',
                                        s=marker_size)

    axes[0].legend()
    axes[0].set_title('Transformed AFM data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the AFM map
    axes[1].plot(median_contour[:, 0], median_contour[:, 1], 'r-', label='Average Contour')
    heatmap_afm = axes[1].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                  np.ma.masked_where(~mask, grid[:, 1]),
                                  c=brillouin_map,
                                  cmap='hot',
                                  s=marker_size,
                                  vmin=vmin,
                                  vmax=vmax)

    axes[1].legend()
    axes[1].set_title('Transformed Brillouin data map')
    axes[1].grid()
    axes[1].axis('equal')

    # Plot colorbars
    cbar = fig.colorbar(heatmap_brillouin, ax=axes[0])
    cbar.set_label('Reduced elastic modulus (Pa)')

    cbar = fig.colorbar(heatmap_afm, ax=axes[1])
    cbar.set_label('Brillouin shift (GHz)')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig
