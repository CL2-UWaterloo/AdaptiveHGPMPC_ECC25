import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import KDEpy as kdepy

from common.plotting_utils import AxisAdjuster
from consts import FIG_SAVE_BOOL


class OnlineKDE_Vanilla:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = kdepy.NaiveKDE(bw=bandwidth)
        self.data = None

    def update(self, new_data):
        if self.data is None:
            self.data = new_data
        else:
            self.data = np.vstack((self.data, new_data))
        self.kde.fit(self.data)

    def estimate_density(self, samples):
        density = self.kde.evaluate(samples)
        return density

    def compute_alpha_map(self, test_samples, fine_ctrl=100,
                          x_min=0, x_max=4, y_min=0, y_max=4,
                          alpha_min=0.1, alpha_max=0.4, OKDE_lower_cutoff=0.125, OKDE_upper_cutoff=0.4, viz=True,
                          fig_save_name='test'):
        density_values = self.estimate_density(test_samples).reshape(fine_ctrl, fine_ctrl)
        OKDE_to_alpha_map = lambda x: interpolate(x, OKDE_lower_cutoff, OKDE_upper_cutoff, alpha_min, alpha_max)
        if viz:
            alpha_map = OKDE_to_alpha_map(density_values)
            viz_OKDE(density_values, x_min, x_max, y_min, y_max)
            viz_OKDE(alpha_map, x_min, x_max, y_min, y_max, fig_save_file=fig_save_name)
        return OKDE_to_alpha_map, OKDE_to_alpha_map(density_values)


def update_and_estimate_density(OKDE_inst, test_points, test_fine_ctrl, data, viz=True,
                                x_min=None, x_max=None, y_min=None, y_max=None, log_norm_viz=False,
                                fig_save_file='test'):
    OKDE_inst.update(data)
    # Estimate density on meshgrid
    density_estimate = OKDE_inst.estimate_density(test_points).reshape(test_fine_ctrl, test_fine_ctrl)
    if viz:
        ax = viz_OKDE(density_estimate, x_min, x_max, y_min, y_max, data, log_norm=log_norm_viz, fig_save_file=fig_save_file)

    return density_estimate


def viz_OKDE(density_estimate, x_min, x_max, y_min, y_max, data=None, log_norm=False, title='', fig_save_file='test', include_colourbar=True, ax=None):
    # Plot heatmap
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    heatmap = ax.imshow(density_estimate, origin='lower', cmap='hot', extent=(x_min, x_max, y_min, y_max),
                         norm=(None if log_norm is False else LogNorm()))
    if include_colourbar:
        # cbar = plt.colorbar(heatmap, label='Density')
        # Add a colorbar to the axis
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.set_label(label='KDE magnitude', size=25)
        cbar.ax.tick_params(labelsize=25)
    if data is not None:
        ax.scatter(data[:, 0], data[:, 1], c='cyan', s=40, label='Data')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    AxisAdjuster(labelsize=25).adjust_ax_object(ax=ax,
                                                title_text="Estimated Density Heatmap" if not title else title,
                                                xlabel_text='x-coord (m)',
                                                ylabel_text='z-coord (m)', set_equal=False, skip_legend=False, legend_loc="lower right")
    if FIG_SAVE_BOOL:
        plt.savefig(fig_save_file+'.svg', format='svg', dpi=300, bbox_inches='tight')
    return ax


def viz_vanilla_OKDE(data=None, kernel_type='gaussian', bandwidth=0.5, log_norm_viz=False, test_fine_ctrl=100, fig_save_file='test', viz=True):
    x_min, x_max = 0, 4
    y_min, y_max = 0, 4

    # Create meshgrid for visualization
    x_range = np.linspace(x_min, x_max, test_fine_ctrl)
    y_range = np.linspace(y_min, y_max, test_fine_ctrl)
    # Create a grid of x[1] and x[2] values
    meshes = np.meshgrid(*[x_range, y_range], indexing='xy')

    # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
    test_points = np.vstack([m.flatten() for m in meshes]).T
    # print(test_points.shape)

    # Initialize OnlineKDE2D
    okde_2d = OnlineKDE_Vanilla(bandwidth=bandwidth, kernel=kernel_type)

    # Streaming update
    density_estimate = update_and_estimate_density(okde_2d, test_points, test_fine_ctrl, data.T, viz=viz,
                                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, log_norm_viz=log_norm_viz,
                                                   fig_save_file=fig_save_file)

    return okde_2d, density_estimate


def test_vanilla_OKDE(samples, bandwidth=0.2, min_cutoff_sigma=1, max_cutoff_sigma=1, max_cutoff_num=2,
                      ret_min_max_cutoffs=False, viz=True, KDE_fig_save_file='test', KDE_above_min_save_file='test', KDE_above_max_save_file='test'):
    okde_inst, density_estimate = viz_vanilla_OKDE(data=samples, bandwidth=bandwidth, fig_save_file=KDE_fig_save_file, viz=viz)

    peak_val = compute_kde_magnitudes(bw=bandwidth, N=samples.shape[1], sigma=0)
    min_sigma_val = compute_kde_magnitudes(bw=bandwidth, N=samples.shape[1], sigma=min_cutoff_sigma)
    # print("Peak %s, min_sigma_val %s, peak-sigma %s" % (peak_val, min_sigma_val, peak_val-min_sigma_val))
    min_cutoff_val = min_sigma_val
    if viz:
        above_min_arr = np.ones_like(density_estimate.ravel())
        min_cutoff_condn = lambda x: np.where(x >= min_sigma_val)[0]
        above_min_arr[min_cutoff_condn(density_estimate.ravel())] = 0
        above_min_arr = above_min_arr.reshape(*density_estimate.shape)
        viz_OKDE(above_min_arr, 0, 4, 0, 4, samples.T, log_norm=False, fig_save_file=KDE_above_min_save_file, include_colourbar=False)

    max_sigma_val = compute_kde_magnitudes(bw=bandwidth, N=samples.shape[1], sigma=max_cutoff_sigma)
    # print("Peak %s, max_sigma_val %s, peak + %s *sigma %s" % (peak_val, max_sigma_val, max_cutoff_num, peak_val+max_cutoff_num*max_sigma_val))
    max_cutoff_val = peak_val + (max_cutoff_num*max_sigma_val)
    if viz:
        above_max_arr = np.zeros_like(density_estimate.ravel())
        max_cutoff_condn = lambda x: np.where(x >= max_cutoff_val)[0]
        above_max_arr[max_cutoff_condn(density_estimate.ravel())] = 1
        above_max_arr = above_max_arr.reshape(*density_estimate.shape)
        ax = viz_OKDE(above_max_arr, 0, 4, 0, 4, samples.T, log_norm=False, fig_save_file=KDE_above_max_save_file, include_colourbar=False)

    if not ret_min_max_cutoffs:
        return okde_inst
    else:
        return okde_inst, min_cutoff_val, max_cutoff_val


def compute_kde_magnitudes(bw, N, sigma=0):
    div_factor = (N * bw)
    if sigma == 0:
        peak_val = 1/(np.sqrt(2*math.pi)) * (1 / div_factor)
        mag_val = peak_val
    else:
        sigma_val = (1/(np.sqrt(2*math.pi)) * np.exp(-sigma/2)) * (1 / div_factor)
        mag_val = sigma_val
    return mag_val


def interpolate(x, x_min_cutoff, x_max_cutoff, k_min, k_max):
    x_shape = x.shape
    x = x.ravel()
    mask1 = x < x_min_cutoff
    mask2 = x > x_max_cutoff

    result = np.where(mask1, k_max, x)
    result = np.where(mask2, k_min, result)

    slope = (k_min - k_max) / (x_max_cutoff - x_min_cutoff)
    result = np.where(~mask1 & ~mask2, k_max + slope * (x - x_min_cutoff), result)

    result = result.reshape(x_shape)
    return result



