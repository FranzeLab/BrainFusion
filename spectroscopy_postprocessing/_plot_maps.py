import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from ._utils import mask_contour
from shapely.geometry import Polygon, Point
from matplotlib.ticker import LogLocator
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.stats import f
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['svg.fonttype'] = 'none'


def format_p_value(p):
    if p < 0.0001:
        return "p<0.001"
    elif 0.001 <= p < 0.20:
        return f"p={p:.3f}"
    elif p >= 0.20:
        return f"p={p:.2f}"
    else:
        return "Invalid p-value"


def plot_contours(average_contour, template_contour, matched_contours):
    fig = plt.figure(figsize=(8, 8))

    # Plot the template contour
    plt.plot(template_contour[:, 0], template_contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5,
             label='Mask (Template)')
    #plt.scatter(template_contour[0, 0], template_contour[0, 1], color='orange', s=12, label='First contour coordinate')

    # Plot the matched contours (only label the first one)
    for i, contour in enumerate(matched_contours[1:]):
        if i == 0:
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5,
                     label='Masks (Matched)')
            #plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)
        else:
            plt.plot(contour[:, 0], contour[:, 1], color='grey', linestyle='-', linewidth=1.5, alpha=0.5)  # No label for subsequent contours
            #plt.scatter(contour[0, 0], contour[0, 1], color='orange', s=12)

    # Plot the median contour
    plt.plot(average_contour[:, 0], average_contour[:, 1], color='blue', linestyle='--', linewidth=4, label='Median Contour')

    plt.legend()
    plt.axis('equal')

    return fig


def plot_maps_on_image(img, data, grid, contour, folder_name, scale=1, label='Brillouin shift (GHz)', cmap='viridis', marker_size=15,
                       vmin=None, vmax=None, log=False, mask=False, alpha=0.9):
    if mask:
        polygon = Polygon(contour)
        inside_points = []
        inside_data = []

        for i, point in enumerate(grid):
            if polygon.contains(Point(point)):
                inside_points.append(point)
                inside_data.append(data[i])

        grid = np.array(inside_points)
        data = np.array(inside_data)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Scale image to µm
    height_in_mu = img.shape[0] / scale
    width_in_mu = img.shape[1] / scale

    # Plot background image and heatmap
    ax.imshow(img, cmap='gray', aspect='equal', origin='lower',
              extent=[0, width_in_mu, 0, height_in_mu], alpha=alpha)

    # Use logarithmic normalization
    if vmin or vmin:
        norm = None
    else:
        norm = mcolors.LogNorm() if log else None

    heatmap = ax.scatter(grid[:, 0],
                         grid[:, 1],
                         c=data,
                         cmap=cmap,
                         s=marker_size,
                         marker='s',
                         alpha=1,
                         vmin=vmin,
                         vmax=vmax,
                         norm=norm
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
                   marker_size=30, vmin=None, vmax=None, mask=False):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the deformed grid using transformed grid values
    # Contours
    axes[0].plot(original_contour[:, 0], original_contour[:, 1], color='grey', linestyle='-', linewidth=3,
                 label='Original Contour')
    axes[0].plot(deformed_contour[:, 0], deformed_contour[:, 1], color='blue', linestyle='--', linewidth=4,
                 label='Deformed Contour')
    #axes[0].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original grid
    if mask:
        polygon = Polygon(original_contour)
        inside_points = []
        inside_data = []

        for i, point in enumerate(bm_grid_points):
            if polygon.contains(Point(point)):
                inside_points.append(point)
                inside_data.append(bm_data[i])

        bm_grid_points = np.array(inside_points)
        bm_data_mask = np.array(inside_data)
    else:
        bm_data_mask = bm_data
    axes[0].scatter(bm_grid_points[:, 0],
                    bm_grid_points[:, 1],
                    c=bm_data_mask,
                    cmap=cmap,
                    s=marker_size,
                    label='Original Grid',
                    vmin=vmin,
                    vmax=vmax)

    #axes[0].legend()
    #axes[0].set_title('Original data map')
    #axes[0].grid()
    axes[0].axis('equal')

    # Plot the deformed grid using new coordinates and transformed data values (griddata)
    # Contours
    axes[1].plot(original_contour[:, 0], original_contour[:, 1], 'grey', linestyle='-', linewidth=3,
                 label='Original Contour')
    axes[1].plot(deformed_contour[:, 0], deformed_contour[:, 1], 'blue', linestyle='--', linewidth=4,
                 label='Deformed Contour')
    #axes[1].plot(trafo_contour[:, 0], trafo_contour[:, 1], 'g--', label='Transformed Original Contour')

    # Original and deformed data
    if mask:
        polygon = Polygon(deformed_contour)
        inside_points = []
        inside_data = []

        for i, point in enumerate(grid_points_trafo):
            if polygon.contains(Point(point)):
                inside_points.append(point)
                inside_data.append(bm_data[i])

        grid_points_trafo = np.array(inside_points)
        bm_data_mask = np.array(inside_data)
    else:
        bm_data_mask = bm_data

    heatmap_trafo_coords = axes[1].scatter(grid_points_trafo[:, 0],
                                           grid_points_trafo[:, 1],
                                           c=bm_data_mask,
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

    #axes[1].legend()
    #axes[1].set_title('Transformed data map')
    #axes[1].grid()
    axes[1].axis('equal')

    # Set axis limits to be the same for both plots
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())

    # Plot colorbar
    cbar = fig.colorbar(heatmap_trafo_coords, ax=axes[1])
    cbar.set_label(label)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig


def plot_average_heatmap(data_maps, grid_trafos, data_avg, grid_avg, average_contour, data_variable,
                         label='Brillouin shift (GHz)', cmap='viridis', marker_size=15, vmin=None, vmax=None,
                         mask=True, interpolate=True):
    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Overlay all heatmaps on top of each other
    # Median contour
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'b--', linewidth=5, label='Median Contour')

    # Heatmaps
    for i, _ in enumerate(data_maps[1:]):
        assert type(mask) is bool, print('The provided mask argument is not boolean!')
        if mask is True:
            mask_1 = mask_contour(average_contour, grid_trafos[i])
        else:
            mask_1 = np.full(grid_trafos[i].shape[0], True)

        if i == 0:
            axes[0].scatter(np.ma.masked_where(~mask_1, grid_trafos[i][:, 0]),
                            np.ma.masked_where(~mask_1, grid_trafos[i][:, 1]),
                            c=data_maps[i][data_variable],
                            cmap=cmap,
                            s=marker_size,
                            label='Transformed Grid',
                            alpha=0.5,
                            vmin=vmin,
                            vmax=vmax)
        else:
            axes[0].scatter(np.ma.masked_where(~mask_1, grid_trafos[i][:, 0]),
                            np.ma.masked_where(~mask_1, grid_trafos[i][:, 1]),
                            c=data_maps[i][data_variable],
                            cmap=cmap,
                            s=marker_size,
                            alpha=0.5,
                            vmin=vmin,
                            vmax=vmax)

    #axes[0].legend()
    #axes[0].set_title('Layered data maps of transformed spatial maps')
    #axes[0].grid()
    axes[0].axis('equal')

    # Average heatmap
    # Median contour
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'b--', linewidth=5, label='Median Contour')

    if interpolate is True:
        # Create grid
        x_min, x_max = min(g[:, 0].min() for g in grid_trafos), max(g[:, 0].max() for g in grid_trafos)
        y_min, y_max = min(g[:, 1].min() for g in grid_trafos), max(g[:, 1].max() for g in grid_trafos)
        grid_x, grid_y = np.mgrid[x_min:x_max:25j, y_min:y_max:40j]  # 20x40 grid for AFM
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        interp_list = []
        for i, _ in enumerate(data_maps[1:]):
            grid = grid_trafos[i][:, :],
            values = data_maps[i][data_variable]
            val_interp = griddata(grid, values, grid_points, method='nearest')
            interp_list.append(val_interp)

        # Convert to NumPy array and compute the mean
        interp_lists = np.array(interp_list)
        mean_data = np.nanmean(interp_list, axis=0)  # Average across all interpolated grids

        # Mask average data
        assert type(mask) is bool, print('The provided mask argument is not boolean!')
        if mask is True:
            mask_2 = mask_contour(average_contour, grid_points)
        else:
            mask_2 = np.full(grid_points.shape[0], True)

        # Average data
        heatmap = axes[1].scatter(np.ma.masked_where(~mask_2, grid_points[:, 0]),
                                  np.ma.masked_where(~mask_2, grid_points[:, 1]),
                                  c=mean_data,
                                  cmap=cmap,
                                  s=marker_size,
                                  marker='s',
                                  label='Transformed Grid',
                                  alpha=1,
                                  vmin=vmin,
                                  vmax=vmax)

        # axes[1].legend()
        # axes[1].set_title('Average data map of transformed spatial maps')
        # axes[1].grid()
        axes[1].axis('equal')

        # Add colorbar
        cbar = fig.colorbar(heatmap, ax=axes[1])
        cbar.set_label(label)

        # Adjust layout to avoid overlap
        plt.tight_layout()
    else:
        # Mask average data
        assert type(mask) is bool, print('The provided mask argument is not boolean!')
        if mask is True:
            for i, _ in enumerate(data_maps[1:]):
                mask_3 = mask_contour(average_contour, grid_avg)
        else:
            mask_3 = np.full(grid_avg.shape[0], True)

        # Average data
        heatmap = axes[1].scatter(np.ma.masked_where(~mask_3, grid_avg[:, 0]),
                                  np.ma.masked_where(~mask_3, grid_avg[:, 1]),
                                  c=data_avg[f'{data_variable}_median'],
                                  cmap=cmap,
                                  s=marker_size,
                                  marker='s',
                                  label='Transformed Grid',
                                  alpha=1,
                                  vmin=vmin,
                                  vmax=vmax)

        #axes[1].legend()
        #axes[1].set_title('Average data map of transformed spatial maps')
        #axes[1].grid()
        axes[1].axis('equal')

        # Add colorbar
        cbar = fig.colorbar(heatmap, ax=axes[1])
        cbar.set_label(label)

        # Adjust layout to avoid overlap
        plt.tight_layout()

    return fig


def plot_corr_maps(average_contour, afm_map, brillouin_map, grid, mask=True, marker_size=40, vmin=None, vmax=None):
    # Mask average data
    assert type(mask) is bool, print('The provided mask argument is not boolean!')
    if mask is True:
        mask = mask_contour(average_contour, grid)
    else:
        mask = np.full(grid.shape[0], True)

    # Create subplots with two side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the AFM map
    axes[0].plot(average_contour[:, 0], average_contour[:, 1], 'r-', label='Average Contour')
    heatmap_brillouin = axes[0].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                        np.ma.masked_where(~mask, grid[:, 1]),
                                        c=afm_map,
                                        marker='s',
                                        cmap='hot',
                                        s=marker_size)

    axes[0].legend()
    axes[0].set_title('Transformed AFM data map')
    axes[0].grid()
    axes[0].axis('equal')

    # Plot the AFM map
    axes[1].plot(average_contour[:, 0], average_contour[:, 1], 'r-', label='Average Contour')
    heatmap_afm = axes[1].scatter(np.ma.masked_where(~mask, grid[:, 0]),
                                  np.ma.masked_where(~mask, grid[:, 1]),
                                  c=brillouin_map,
                                  marker='s',
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


def plot_norm_corr(map1, map2, pearson=None, p_value=None, label1="X-axis", label2="Y-axis",
                   output_path=None, square_root_fit=False, map1_fit_limits=None, map2_fit_limits=None):
    # Ensure map1 and map2 are arrays
    map1, map2 = np.asarray(map1), np.asarray(map2)
    if map1.shape != map2.shape:
        raise ValueError("map1 and map2 must have the same shape.")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(map1, map2, color='blue', s=10, label='Normalized data')

    # Filter for fitting
    mask1 = mask2 = np.full(map1.shape, True, dtype=bool)
    if map1_fit_limits:
        mask1 = (map1 > map1_fit_limits[0]) & (map1 < map1_fit_limits[1])

    if map2_fit_limits:
        mask2 = (map2 > map2_fit_limits[0]) & (map2 < map2_fit_limits[1])

    # Apply the combined mask
    combined_mask = mask1 & mask2
    map1, map2 = map1[combined_mask], map2[combined_mask]

    if square_root_fit:
        def sqrt_model(x, a, b):
            return a * np.sqrt(x) + b

        try:
            # Fit the square-root model
            params, _ = curve_fit(sqrt_model, map1, map2)
            a, b = params

            # Generate fitted values
            x_fit = np.linspace(min(map1), max(map1), 100)
            y_fit = sqrt_model(x_fit, *params)

            # Evaluate fit quality
            y_pred = sqrt_model(map1, *params)
            rss_alt = np.sum((map2 - y_pred) ** 2)
            y_mean = np.mean(map2)
            tss = np.sum((map2 - y_mean) ** 2)
            r_squared = 1 - (rss_alt / tss)
            rss_null = tss
            df_model = len(params) - 1
            df_residual = len(map2) - len(params)
            f_stat = ((rss_null - rss_alt) / df_model) / (rss_alt / df_residual)
            p_value = 1 - f.cdf(f_stat, df_model, df_residual)

            # Plot the fit
            ax.plot(x_fit, y_fit, color='red', label=r'Fit: $y = a\sqrt{x + b} + c$')

            # Annotate with R^2 and p-value
            pval = format_p_value(p_value)
            ax.text(0.60 - len(pval) / 200, 0.88, f'$\\mathbf{{R^2}}$: {np.round(r_squared, 3)}\np-value: {pval}',
                    fontsize=12, fontweight='bold', color='red', ha='left', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))
        except RuntimeError:
            ax.text(0.5, 0.9, 'Fit failed', fontsize=12, color='red', ha='center', transform=ax.transAxes)
            r_squared = None

    else:
        # Add Pearson correlation and p-value
        if pearson is not None and p_value is not None:
            pval = format_p_value(p_value)
            ax.text(0.60 - len(pval) / 200, 0.88, f'Pearson: {np.round(pearson, 3)}\np-value: {pval}',
                    fontsize=12, fontweight='bold', color='red', ha='left', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='red', boxstyle='round,pad=0.5'))

    # Customize plot
    ax.set_xlabel(label1, fontsize=18)
    ax.set_ylabel(label2, fontsize=18)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tick_params(axis='both', which='major', length=4, width=1.5)
    plt.axis('equal')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    if square_root_fit:
        ax.legend(loc='upper left')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_cumulative(analysis_file, raw_key, projected=True, results_folder=None, bin=0.01, label='', log=False,
                    loc='upper left', x_ticks=6, round=1):
    results_folder = results_folder or './figures_for_thesis'

    # Prepare bins
    all_values = np.concatenate([
        exp_value['raw_data'][f'{raw_key}_distribution']
        for exp_key, exp_value in analysis_file.items()
        if '#' in exp_key
    ])
    lower, upper = np.nanpercentile(all_values, 1), np.nanpercentile(all_values, 99)
    bins = np.arange(lower, upper + bin, bin)

    raw_results = []
    projected_results = []

    for exp_key, exp_value in analysis_file.items():
        if '#' in exp_key:
            # Raw data
            raw_data = exp_value['raw_data'][f'{raw_key}_distribution']
            dist, _ = np.histogram(raw_data, bins=bins)
            ndist = dist / sum(dist)
            cdist = np.cumsum(ndist) * 100
            raw_results.append(cdist)

            # Projected data
            if projected:
                proj_data = exp_value['raw_data'][f'{raw_key}_proj']
                proj_data_flat = np.sort(proj_data.flatten())
                proj_dist, _ = np.histogram(proj_data_flat, bins=bins)
                proj_ndist = proj_dist / sum(proj_dist)
                proj_cdist = np.cumsum(proj_ndist) * 100
                projected_results.append(proj_cdist)

    # Averaging raw results
    raw_cdist_aver = np.nanmean(raw_results, axis=0)
    raw_cdist_std = np.nanstd(raw_results, axis=0)
    raw_lower_bound = raw_cdist_aver - raw_cdist_std
    raw_upper_bound = raw_cdist_aver + raw_cdist_std

    # Averaging projected results if needed
    if projected:
        proj_cdist_aver = np.nanmean(projected_results, axis=0)
        proj_cdist_std = np.nanstd(projected_results, axis=0)
        proj_lower_bound = proj_cdist_aver - proj_cdist_std
        proj_upper_bound = proj_cdist_aver + proj_cdist_std

    # Plot cumulative distribution
    plt.figure(figsize=(4, 6))
    # Raw data plot
    plt.plot(bins[:-1], raw_cdist_aver, color='black', linewidth=3, label='3D dataset', linestyle='-')
    plt.fill_between(bins[:-1], raw_lower_bound, raw_upper_bound, color='#A9A9A9', alpha=0.4, edgecolor='none')

    # Projected data plot
    if projected:
        plt.plot(bins[:-1], proj_cdist_aver, color='blue', linewidth=3, label='Median z-axis projection',
                 linestyle='-')
        plt.fill_between(bins[:-1], proj_lower_bound, proj_upper_bound, color='#1E90FF', alpha=0.4, edgecolor='none')

    # Customize plot labels and title
    plt.xlabel(label, fontsize=21)
    plt.ylabel('Cumulative Percentage (%)', fontsize=21)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tick_params(axis='both', which='major', length=4, width=1.5)
    plt.ylim([0, 100])

    plt.xlim([lower, upper])
    plt.xticks(np.round(np.linspace(lower, upper, x_ticks), round), fontsize=14)

    if log:
        plt.xscale('log')

    if loc is not None:
        plt.legend(loc=loc, fontsize=12)

    # Save and show the plot
    save_path = os.path.join(results_folder, f'CumulativePercentage_{raw_key}.svg')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_experiments(analysis_file, results_folder, raw_data_key, label='', cmap='viridis', marker_size=20, vmin=None,
                     vmax=None, **kwargs):
    # Plot regular heatmaps
    average_contour = analysis_file['average_contour']
    avg_data = analysis_file['average_data']
    avg_grid = analysis_file['average_grid']
    extended_grid = analysis_file['extended_grid']

    """
    for exp_key, exp_value in analysis_file.items():
        if '#' in exp_key:
            fig = plot_maps_on_image(exp_value['brightfield_image'],
                                     exp_value['raw_data'][f'{raw_data_key}'],
                                     exp_value['raw_grid'],
                                     exp_value['contour'],
                                     exp_key,
                                     scale=exp_value['pix_per_um'],
                                     label=label,
                                     cmap=cmap,
                                     marker_size=marker_size,
                                     vmin=vmin,
                                     vmax=vmax,
                                     log=False,
                                     mask=True)
            output_path = os.path.join(results_folder, f'{exp_key}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Plot transformed heatmap and contours
            fig = plot_cont_func(exp_value['contour'],
                                 average_contour,
                                 exp_value['trafo_contour'],
                                 exp_value['raw_data'][f'{raw_data_key}'],
                                 exp_value['trafo_data'][f'{raw_data_key}_trafo'],
                                 exp_value['raw_grid'],
                                 exp_value['trafo_grid'],
                                 extended_grid,
                                 label=label,
                                 cmap=cmap,
                                 marker_size=120,
                                 vmin=vmin,
                                 vmax=vmax,
                                 mask=True)
            output_path = os.path.join(results_folder, f'MeanContourTransformed_{exp_key}.svg')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    """

    # Plot all transformed heatmaps and the averaged heatmap
    data_list = [value['raw_data'] for key, value in analysis_file.items() if '#' in key]
    grid_trafo_list = [value['trafo_grid'] for key, value in analysis_file.items() if '#' in key]

    fig = plot_average_heatmap(data_list,
                               grid_trafo_list,
                               avg_data,
                               avg_grid,
                               average_contour,
                               f'{raw_data_key}',
                               label=label,
                               cmap=cmap,
                               marker_size=120,
                               vmin=vmin,
                               vmax=vmax)
    output_path = os.path.join(results_folder, f'Averaged_{raw_data_key.capitalize()}_Maps.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
