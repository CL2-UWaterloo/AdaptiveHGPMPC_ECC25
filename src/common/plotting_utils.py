import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox

from .box_constraint_utils import box_constraint_direct


def plot_uncertainty_bounds_1d(observed_pred, region_x, ax, colours, idx, custom_text=None):
    lower, upper = observed_pred.confidence_region()
    ax.plot(region_x.squeeze(), observed_pred.mean.numpy(), colours[idx], label='GP mode %s' % (idx+1) if custom_text is None else custom_text)
    ax.fill_between(region_x.squeeze(), lower.numpy(), upper.numpy(), alpha=0.5)


def plot_constraint_sets(plot_idxs, inp_type="shrunk_vec", alpha=0.8, colour='r', ax=None, **kwargs):
    if inp_type == "shrunk_vec":
        reqd_keys = ["shrunk_vec"]
        assert all([key in kwargs for key in reqd_keys]), "Missing required keys for plotting shrunk vec representation"
        box = box_constraint_direct(kwargs["shrunk_vec"], skip_bound_construction=False, plot_idxs=plot_idxs)
    else:
        raise NotImplementedError
    box.plot_constraint_set(ax=ax, alpha=alpha, colour=colour)


def save_fig(axes, fig_name, tick_sizes, tick_skip=1, k_range=None):
    for ax in axes:
        ax.tick_params(axis='both', labelsize=tick_sizes)
        if k_range is not None:
            ax.set_xticks(k_range[::tick_skip])
        ax.legend(prop=dict(size=tick_sizes))
    plt.savefig(fig_name+'.svg', format='svg', dpi=300)


def copy_old_ax_to_new_ax(old_ax, new_ax, fontsize_override=None):
    lines = old_ax.get_lines()  # Get all line objects
    scatter_plots = [child for child in old_ax.get_children() if isinstance(child, PathCollection)]
    for line in lines:
        new_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color())

    # Recreate scatter plots
    for scatter in scatter_plots:
        offsets = scatter.get_offsets()
        sizes = scatter.get_sizes()
        colors = scatter.get_facecolors()
        new_ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=colors, label=scatter.get_label())

    for child in old_ax.get_children():
        if isinstance(child, AxesImage):
            image = child
            break
    if image is not None:
        extent = image.get_extent()
        origin = image.origin
        new_ax.imshow(image.get_array(), cmap=image.get_cmap(), norm=image.norm, extent=extent, origin=origin)

    title = old_ax.get_title()
    xlabel = old_ax.get_xlabel()
    ylabel = old_ax.get_ylabel()
    legend_labels = [label.get_text() for label in old_ax.get_legend().get_texts()] if old_ax.get_legend() else []
    new_ax.set_title(title)  # Set the title text first
    new_ax.set_xlabel(xlabel)
    new_ax.set_ylabel(ylabel)
    if legend_labels:
        new_ax.legend(legend_labels)

    copy_ax_properties(base_ax=old_ax, ax_to_copy_to=new_ax, fontsize_override=fontsize_override)


def copy_ax_properties(base_ax, ax_to_copy_to, fontsize_override=None):
    ax_to_copy_to = ax_to_copy_to
    font_sizes = {
        "title":  base_ax.title.get_fontsize() if base_ax.title else None,
        "xlabel": base_ax.xaxis.label.get_fontsize() if base_ax.xaxis.label else None,
        "ylabel": base_ax.yaxis.label.get_fontsize() if base_ax.yaxis.label else None,
        "legend": base_ax.get_legend().get_fontsize() if base_ax.get_legend() else None,
        "xticks": base_ax.get_xticklabels()[0].get_fontsize() if base_ax.get_xticklabels() else None,
        "yticks": base_ax.get_yticklabels()[0].get_fontsize() if base_ax.get_yticklabels() else None
    }
    ax_to_copy_to.title.set_fontsize(font_sizes["title"] if fontsize_override is None else fontsize_override)  # Set the title font size
    ax_to_copy_to.xaxis.label.set_fontsize(font_sizes["xlabel"] if fontsize_override is None else fontsize_override)  # Set the x-label font size
    ax_to_copy_to.yaxis.label.set_fontsize(font_sizes["ylabel"] if fontsize_override is None else fontsize_override)  # Set the y-label font size

    if font_sizes["legend"]:
        ax_to_copy_to.get_legend().set_fontsize(font_sizes["legend"] if fontsize_override is None else fontsize_override)  # Set the legend font size
    if font_sizes["xticks"]:
        ax_to_copy_to.tick_params(axis='x', labelsize=font_sizes["xticks"] if fontsize_override is None else fontsize_override)  # Set x-tick font size
    if font_sizes["yticks"]:
        ax_to_copy_to.tick_params(axis='y', labelsize=font_sizes["yticks"] if fontsize_override is None else fontsize_override)  # Set y-tick font size



def save_fig_plt(fig_name, tick_sizes, tick_skip=1, k_range=None):
    plt.tick_params(axis='both', labelsize=tick_sizes)
    # plt.ticklabel_format(axis='both', style='plain')
    if k_range is not None:
        plt.xticks(k_range[::tick_skip])
    plt.savefig(fig_name+'.svg', format='svg', dpi=300)
    plt.show()


def generate_fine_grid(start_limit, end_limit, fineness_param, viz_grid=False, num_samples=200):
    # print(start_limit[0, :])
    # print(start_limit, start_limit.shape[-1])
    fine_coords_arrs = [torch.linspace(start_limit[idx, :].item(), end_limit[idx, :].item(),
                                       (int(fineness_param[idx]*(end_limit[idx, :]-start_limit[idx, :]).item()) if fineness_param is not None else num_samples))
                        for idx in range(start_limit.shape[0])]
    meshed = np.meshgrid(*fine_coords_arrs)
    grid_vectors = np.vstack([mesh.flatten() for mesh in meshed]) # Shape = dims * num_samples where num_samples is controller by fineness_param and start and end lims
    # print(grid_vectors.shape)

    if viz_grid:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(*[grid_vectors[axis_idx, :] for axis_idx in range(grid_vectors.shape[0])], c='b')
    return grid_vectors


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def dir_exist_or_create(base_path, sub_path=None):
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    store_dir = base_path
    if sub_path is not None:
        if not os.path.exists(base_path+sub_path):
            os.mkdir(base_path+sub_path)
        store_dir = base_path+sub_path
    return store_dir


def save_subplot(ax, figure, fig_name=None, base_path="C:\\Users\\l8souza\\PycharmProjects\\GP4MPC\\src\\images\\",
                 sub_path="scalar_motivating_example\\", extension='.svg'):
    assert fig_name is not None, "Need to set the fig_name attribute"
    # Save just the portion _inside_ the second axis's boundaries
    extent = full_extent(ax).transformed(figure.dpi_scale_trans.inverted())
    store_dir = dir_exist_or_create(base_path, sub_path=sub_path)
    figure.savefig(store_dir+fig_name+extension, bbox_inches=extent)


class AxisAdjuster:
    def __init__(self, labelsize=18):
        self.labelsize = labelsize

    def adjust_ax_object(self, ax, title_text, xlabel_text, ylabel_text, set_equal=False, skip_legend=False, legend_loc=None, tickwidth=None, ticklength=None):
        tick_params_addn = {}
        if tickwidth is not None:
            tick_params_addn = {"width": tickwidth, "length": ticklength}
        ax.tick_params(axis='both', labelsize=self.labelsize, **tick_params_addn)
        ax.set_title(title_text, fontsize=self.labelsize)
        ax.set_xlabel(xlabel_text, fontsize=self.labelsize)
        ax.set_ylabel(ylabel_text, fontsize=self.labelsize)
        if not skip_legend:
            if legend_loc is not None:
                ax.legend(fontsize=self.labelsize, loc=legend_loc)
            else:
                ax.legend(fontsize=self.labelsize)
        if set_equal:
            ax.axis('equal')
