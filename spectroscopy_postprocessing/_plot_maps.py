import matplotlib.pyplot as plt
import numpy as np
from _utilis import mask_contour, scatter_with_touching_squares


def plot_maps_on_image(img, data, grid, folder_name, label='Brillouin shift (GHz)', cmap='viridis', marker_size=15,
                       vmax=None, vmin=None):
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
                         vmax=vmin,
                         vmin=vmax
                         )

    ax.set_title(f'{folder_name}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(label)

    return fig


def plot_cont_func(original_contour, deformed_contour, trafo_contour, bm_data, bm_data_trafo_list,
                   bm_grid_points, grid_points_trafo, extended_grid):
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
                    cmap='viridis',
                    s=15,
                    label='Original Grid',
                    alpha=1)

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
    axes[1].scatter(grid_points_trafo[:, 0],
                    grid_points_trafo[:, 1],
                    c=bm_data,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (New coordinates)',
                    alpha=1)
    axes[1].scatter(extended_grid[:, 0],
                    extended_grid[:, 1],
                    c=bm_data_trafo_list,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (Griddata)',
                    alpha=0)

    axes[1].legend()
    axes[1].set_title('Transformed data map')
    axes[1].grid()
    axes[1].axis('equal')

    # axes[0].set_xlim([0, 1023])
    # axes[0].set_ylim([200, 1023])
    # axes[1].set_xlim([0, 1023])
    # axes[1].set_ylim([200, 1023])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_cont_func_afm(original_contour, deformed_contour, trafo_contour, data, grid, data_trafo, grid_trafo):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original grid
    axes[0].scatter(grid[:, 0],
                    grid[:, 1],
                    c=data,
                    cmap='hot',
                    s=15,
                    label='Original Grid',
                    alpha=1)

    axes[0].legend()
    axes[0].set_title('Original AFM data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using new coordinates and transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'r-', label='Deformed Contour')
    axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    axes[1].scatter(grid_trafo[:, 0],
                    grid_trafo[:, 1],
                    c=data,
                    cmap='viridis',
                    s=15,
                    label='Transformed Grid (New coordinates)',
                    alpha=1)
    axes[1].scatter(grid[:, 0],
                    grid[:, 1],
                    c=data_trafo,
                    cmap='viridis',
                    s=1,
                    label='Transformed Grid (Interpolated on regular grid)',
                    alpha=0)

    axes[1].legend()
    axes[1].set_title('Transformed AFM data map')
    axes[1].grid()
    axes[1].axis('equal')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


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


def plot_average_heatmap(data_maps, data_map_avg, grid, average_contour, data_variable, label='Brillouin shift (GHz)',
                         cmap='viridis', marker_size=15, vmax=None, vmin=None, mask=True, distort=2):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Median contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Heatmaps
    for i, data_map in enumerate(data_maps[1:]):
        if i == 0:
            axes[0].scatter(grid[:, 0],
                            grid[:, 1],
                            c=data_map[data_variable],
                            cmap=cmap,
                            s=marker_size,
                            label='Transformed Grid',
                            alpha=0.5,
                            vmax=vmax,
                            vmin=vmin)
        else:
            axes[0].scatter(grid[:, 0] + np.random.randn(grid.shape[0]) * distort,
                            grid[:, 1] + np.random.randn(grid.shape[0]) * distort,
                            c=data_map[data_variable],
                            cmap=cmap,
                            s=marker_size,
                            alpha=0.5,
                            vmax=vmax,
                            vmin=vmin)

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
        mask = mask_contour(average_contour, grid)
    else:
        mask = np.full(grid.shape[0], True)

    # Average data
    heatmap = axes[1].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                              np.ma.masked_where(~mask, grid[:, 1]),
                              c=data_map_avg,
                              cmap=cmap,
                              s=marker_size,
                              marker='s',
                              label='Transformed Grid',
                              alpha=1,
                              vmax=vmax,
                              vmin=vmin)

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
