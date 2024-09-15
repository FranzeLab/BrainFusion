import matplotlib.pyplot as plt
import numpy as np


def average_heatmap(data_maps):
    return np.median(data_maps, axis=0)


def plot_average_heatmap(data_maps, avg_data_map, grids, avg_grid, average_contour):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Median contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Heatmaps
    for i, data_map in enumerate(data_maps[1:]):
        if i == 0:
            axes[0].scatter(grids[i][:, :, 0],
                            grids[i][:, :, 1],
                            c=data_map,
                            cmap='viridis',
                            s=1,
                            label='Transformed Grid',
                            alpha=1)
        else:
            axes[0].scatter(grids[i][:, :, 0],
                            grids[i][:, :, 1],
                            c=data_map,
                            cmap='viridis',
                            s=1,
                            alpha=1)

    axes[0].legend()
    axes[0].set_title('Overlayed Brillouin maps of transformed spatial maps')
    axes[0].grid()
    axes[0].axis('equal')

    # Average heatmap
    # Median contour
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'r--', linewidth=2, label='Median Contour')

    # Original and deformed data
    axes[1].scatter(avg_grid[:, :, 0],
                    avg_grid[:, :, 1],
                    c=avg_data_map,
                    cmap='viridis',
                    s=1,
                    label='Transformed Grid',
                    alpha=1)

    axes[1].legend()
    axes[1].set_title('Average Brillouin shift of transformed spatial maps')
    axes[1].grid()
    axes[1].axis('equal')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig
