from matplotlib.colors import ListedColormap
import copy
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch

from common.plotting_utils import save_fig, AxisAdjuster 
from consts import FIG_SAVE_BOOL


class box_constraint:
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, lb=None, ub=None, plot_idxs=None):
        """
        :param lb: dimwise list of lower bounds.
        :param ub: dimwise list of lower bounds.
        :param plot_idxs: When plotting, the box itself might be defined in some dimension greater than 2 but we might only want to
        plot the workspace variables and so plot_idxs allows us to limit the consideration of plot_constraint_set to those variables.
        """
        self.lb = np.array(lb, ndmin=2).reshape(-1, 1)
        self.ub = np.array(ub, ndmin=2).reshape(-1, 1)
        self.plot_idxs = plot_idxs
        self.dim = self.lb.shape[0]
        assert (self.lb < self.ub).all(), "Lower bounds must be greater than corresponding upper bound for any given dimension, %s, %s" % (self.lb, self.ub)
        self.setup_constraint_matrix()

    def __str__(self): return "Lower bound: %s, Upper bound: %s" % (self.lb, self.ub)

    def get_random_vectors(self, num_samples):
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        dim = self.lb.shape[0]
        # Casadi can't do matrix mult with Torch instances but only numpy instead. So have to use the np version of the H and b matrix/vector when
        # defining constraints in the opti stack.
        self.H_np = np.vstack((-np.eye(dim), np.eye(dim)))
        self.H = torch.Tensor(self.H_np)
        # self.b = torch.Tensor(np.hstack((-self.lb, self.ub)))
        self.b_np = np.vstack((-self.lb, self.ub))
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b
        self.sym_func_arr = lambda x: self.H_np @ x - self.b_np

    def check_satisfaction(self, sample):
        # If sample is within the polytope defined by the constraints return 1 else 0.
        # print(sample, np.array(sample, ndmin=2).T, self.sym_func(sample), self.b)
        return (self.sym_func(sample) <= 0).all()

    def check_satisfaction_arr(self, sample_arr):
        if type(sample_arr) is torch.Tensor:
            sample_arr = sample_arr.numpy()
        return (self.sym_func_arr(sample_arr) <= 0).all(axis=0)

    def generate_uniform_samples(self, num_samples):
        n = int(np.round(num_samples**(1./self.lb.shape[0])))

        # Generate a 1D array of n equally spaced values between the lower and upper bounds for each dimension
        coords = []
        for i in range(self.lb.shape[0]):
            coords.append(np.linspace(self.lb[i, 0], self.ub[i, 0], n))

        # Create a meshgrid of all possible combinations of the n-dimensions
        meshes = np.meshgrid(*coords, indexing='ij')

        # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
        samples = np.vstack([m.flatten() for m in meshes])

        # Truncate the array to K samples
        samples = samples[:num_samples, :]

        # Print the resulting array
        return samples

    def plot_constraint_set(self, ax=None, alpha=0.8, colour='r'):
        if type(self.plot_idxs) is list:
            if len(self.plot_idxs) != 2:
                if self.dim != 2:
                    raise NotImplementedError("Plotting is only possible for 2D box constraints")
                else:
                    self.plot_idxs = [0, 1]
        else:
            if self.dim != 2:
                assert type(self.plot_idxs) is list, "plot_idxs must be a list of indices of the dimensions to plot"
            else:
                self.plot_idxs = [0, 1]

        plot_mat = np.zeros((2, self.dim))
        for row_idx, plot_idx in enumerate(self.plot_idxs):
            plot_mat[row_idx, plot_idx] = 1

        H_np = np.vstack((-np.eye(2), np.eye(2)))
        b_np_upper, b_np_lower = self.b_np[:self.dim, :], self.b_np[self.dim:, :]
        b_np = np.vstack([plot_mat @ b_np_upper, plot_mat @ b_np_lower])

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        poly_temp = pc.Polytope(H_np, b_np)
        vertices = pc.extreme(poly_temp)
        ax.fill(vertices[:, 0], vertices[:, 1], color=colour, alpha=alpha, edgecolor='black')

    def clip_to_bounds(self, samples):
        return np.clip(samples, self.lb, self.ub)


class box_constraint_2d:
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, x_min, x_max, y_min, y_max, plot_idxs=None):
        self.lb = np.array(np.vstack([x_min, y_min]), ndmin=2).reshape(-1, 1)
        self.ub = np.array(np.vstack([x_max, y_max]), ndmin=2).reshape(-1, 1)
        self.plot_idxs = plot_idxs
        self.dim = self.lb.shape[0]
        assert (self.lb < self.ub).all(), "Lower bounds must be greater than corresponding upper bound for any given dimension, %s, %s" % (self.lb, self.ub)
        self.setup_constraint_matrix()

    def __str__(self): return "Lower bound: %s, Upper bound: %s" % (self.lb, self.ub)

    def get_random_vectors(self, num_samples):
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        dim = self.lb.shape[0]
        # Casadi can't do matrix mult with Torch instances but only numpy instead. So have to use the np version of the H and b matrix/vector when
        # defining constraints in the opti stack.
        self.H_np = np.vstack((-np.eye(dim), np.eye(dim)))
        self.H = torch.Tensor(self.H_np)
        # self.b = torch.Tensor(np.hstack((-self.lb, self.ub)))
        self.b_np = np.vstack((-self.lb, self.ub))
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b
        self.sym_func_arr = lambda x: self.H_np @ x - self.b_np

    def check_satisfaction(self, sample):
        # If sample is within the polytope defined by the constraints return 1 else 0.
        # print(sample, np.array(sample, ndmin=2).T, self.sym_func(sample), self.b)
        return (self.sym_func(sample) <= 0).all()

    def check_satisfaction_arr(self, sample_arr):
        if type(sample_arr) is torch.Tensor:
            sample_arr = sample_arr.numpy()
        return (self.sym_func_arr(sample_arr) <= 0).all(axis=0)

    def generate_uniform_samples(self, num_samples=None, samples_per_dim=None):
        if samples_per_dim is None:
            n = int(np.round(num_samples**(1./self.lb.shape[0])))
            n = [n,] * self.lb.shape[0]
        else:
            n = samples_per_dim

        # Generate a 1D array of n equally spaced values between the lower and upper bounds for each dimension
        coords = []
        for i in range(self.lb.shape[0]):
            coords.append(np.linspace(self.lb[i, 0], self.ub[i, 0], n[i]))

        # Create a meshgrid of all possible combinations of the n-dimensions
        meshes = np.meshgrid(*coords, indexing='ij')

        # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
        samples = np.vstack([m.flatten() for m in meshes])

        # Truncate the array to K samples
        if samples_per_dim is None:
            samples = samples[:num_samples, :]

        # Print the resulting array
        return samples

    def plot_constraint_set(self, ax=None, alpha=0.8, colour='r', label=None):
        if type(self.plot_idxs) is list:
            if len(self.plot_idxs) != 2:
                if self.dim != 2:
                    raise NotImplementedError("Plotting is only possible for 2D box constraints")
                else:
                    self.plot_idxs = [0, 1]
        else:
            if self.dim != 2:
                assert type(self.plot_idxs) is list, "plot_idxs must be a list of indices of the dimensions to plot"
            else:
                self.plot_idxs = [0, 1]

        plot_mat = np.zeros((2, self.dim))
        for row_idx, plot_idx in enumerate(self.plot_idxs):
            plot_mat[row_idx, plot_idx] = 1

        H_np = np.vstack((-np.eye(2), np.eye(2)))
        b_np_upper, b_np_lower = self.b_np[:self.dim, :], self.b_np[self.dim:, :]
        b_np = np.vstack([plot_mat @ b_np_upper, plot_mat @ b_np_lower])

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        poly_temp = pc.Polytope(H_np, b_np)
        vertices = pc.extreme(poly_temp)
        ax.fill(vertices[:, 0], vertices[:, 1], color=colour, alpha=alpha, edgecolor='black', label=label)

    def clip_to_bounds(self, samples):
        return np.clip(samples, self.lb, self.ub)


class box_constraint_direct(box_constraint):
    """
    Bounded constraints lb <= x <= ub as polytopic constraints -Ix <= -b and Ix <= b. np.vstack(-I, I) forms the H matrix from III-D-b of the paper
    """
    def __init__(self, b_np, skip_bound_construction=False, plot_idxs=None):
        self.plot_idxs = plot_idxs
        self.b_np = np.array(b_np, ndmin=2).reshape(-1, 1)
        self.dim = self.b_np.shape[0] // 2
        self.H_np = np.vstack((-np.eye(self.dim), np.eye(self.dim)))
        self.skip_bound_construction = skip_bound_construction
        if not skip_bound_construction:
            self.retrieve_ub_lb()
        self.setup_constraint_matrix()

    def retrieve_ub_lb(self):
        lb = -self.b_np[:self.dim]
        ub = self.b_np[self.dim:]
        self.lb = np.array(lb, ndmin=2)
        self.ub = np.array(ub, ndmin=2)

    def __str__(self):
        if not self.skip_bound_construction:
            return "Hx<=b ; H: %s, b: %s" % (self.H_np, self.b_np)
        else:
            super().__str__()

    def get_random_vectors(self, num_samples):
        if self.skip_bound_construction:
            assert NotImplementedError, "You chose to skip bound construction. The sampling method currently implemented" \
                                        "requires bounds to generate random samples."
        rand_samples = np.random.rand(self.dim, num_samples)
        for i in range(self.dim):
            scale_factor, shift_factor = (self.ub[i] - self.lb[i]), self.lb[i]
            rand_samples[i, :] = (rand_samples[i, :] * scale_factor) + shift_factor
        return rand_samples

    def setup_constraint_matrix(self):
        self.H = torch.Tensor(self.H_np)
        self.b = torch.Tensor(self.b_np)
        # print(self.b)
        self.sym_func = lambda x: self.H @ np.array(x, ndmin=2).T - self.b


def combine_box(box1, box2, verbose=False):
    box1_lb, box1_ub = box1.lb, box1.ub
    box2_lb, box2_ub = box2.lb, box2.ub
    new_lb, new_ub = np.vstack((box1_lb, box2_lb)), np.vstack((box1_ub, box2_ub))
    new_constraint = box_constraint(new_lb, new_ub)
    if verbose:
        print(new_constraint)
    return new_constraint


class Box_Environment:
    def __init__(self, num_boxes, num_surf, lb_arr, ub_arr, box2surfid, x_min=0, x_max=4, y_min=0, y_max=4, **kwargs):
        self.num_boxes = num_boxes
        self.num_surf = num_surf
        self.lb_arr = lb_arr
        self.ub_arr = ub_arr
        self.box2surfid = box2surfid
        self.surf2boxid = {}
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        self.invert_box2surf()

        boxes = kwargs.get('boxes', None)
        self.boxes = boxes
        if boxes is None:
            self.boxes = [box_constraint(lb=lb_arr[i], ub=ub_arr[i]) for i in range(self.num_boxes)]

    def invert_box2surf(self):
        for k, v in self.box2surfid.items():
            self.surf2boxid[v] = self.surf2boxid.get(v, []) + [k]

    def assign_samples2surf(self, samples, skip_overlap_check=False):
        box2satisfied_dict = {}
        surf2satisfied_dict = {}
        num_samples = samples.shape[1]
        for box_idx, box in enumerate(self.boxes):
            samples_satisfied = box.check_satisfaction_arr(samples)
            box2satisfied_dict[box_idx] = samples_satisfied

        for box_idx in range(len(self.boxes)):
            surf_idx = self.box2surfid[box_idx]
            samples_satisfied = box2satisfied_dict[box_idx]
            if surf2satisfied_dict.get(surf_idx, None) is None:
                surf2satisfied_dict[surf_idx] = samples_satisfied
            else:
                surf2satisfied_dict[surf_idx] = surf2satisfied_dict[surf_idx] | samples_satisfied
        for surf_idx in range(self.num_surf):
            if surf2satisfied_dict.get(surf_idx, None) is None:
                surf2satisfied_dict[surf_idx] = np.array([False, ]*num_samples)

        # Necessary to prevent a single sample being assigned multiple surfaces when present in an intersection region of
        # 2 boxes with conflicting surface ids.
        if not skip_overlap_check:
            for hp_surf_idx in range(self.num_surf-1, -1, -1):
                for lp_surf_idx in range(hp_surf_idx-1, -1, -1):
                    try:
                        surf2satisfied_dict[lp_surf_idx][np.where(surf2satisfied_dict[hp_surf_idx] == True)[0]] = False
                    except KeyError:
                        continue
        return surf2satisfied_dict

    def gen_test_samples(self, fine_ctrl=100):
        x_range = np.linspace(self.x_min, self.x_max, fine_ctrl)
        y_range = np.linspace(self.y_min, self.y_max, fine_ctrl)
        # Create a grid of x[1] and x[2] values
        meshes = np.meshgrid(*[x_range, y_range], indexing='xy')

        # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
        samples = np.vstack([m.flatten() for m in meshes])
        return samples

    def visualize_env(self, fine_ctrl=100, ax=None, alpha=1, fig_save_name='test',
                      colours=('r', 'g', 'b', 'cyan', 'black')):
        samples = self.gen_test_samples(fine_ctrl)
        _, ax = self.viz_samples(samples, ax=ax, alpha=alpha, fig_save_name=fig_save_name, colours=colours, ret_ax=True)
        return ax

    def assign_samples2surfidx(self, samples, skip_overlap_check=False):
        surf2satisfied_dict = self.assign_samples2surf(samples, skip_overlap_check=skip_overlap_check)
        size = samples.shape[1]

        sample_labels = np.ones(size)*-1
        for surf_idx in range(self.num_surf):
            try:
                sample_labels[np.where(surf2satisfied_dict[surf_idx])[0]] = surf_idx
            except KeyError:
                continue
        return surf2satisfied_dict, sample_labels

    def viz_samples(self, samples, ax=None, scale_for_imshow=False, fine_ctrl=100, alpha=1, fig_save_name='test',
                    colours=('r', 'g', 'b', 'cyan', 'black'), ret_ax=False, title=''):
        surf2satisfied_dict, sample_labels = self.assign_samples2surfidx(samples)
        # if alpha==1:
        #     print(surf2satisfied_dict)
        #     print(samples[0])
        #     print(samples[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_xlim(self.x_min, self.x_max)
        if scale_for_imshow:
            samples = self.scale_samples(samples, fine_ctrl)
        sc = ax.scatter(samples[0], samples[1],
                        c=sample_labels, cmap=ListedColormap(['r', 'g', 'b']), vmin=0, vmax=self.num_surf-1, alpha=alpha)
        AxisAdjuster(labelsize=25).adjust_ax_object(ax=ax,
                                                    title_text="Environment mode distribution" if not title else title,
                                                    xlabel_text='x-coord (m)',
                                                    ylabel_text='y-coord (m)', set_equal=False, skip_legend=True, legend_loc="lower right")
        if FIG_SAVE_BOOL:
            save_fig(axes=[ax], fig_name=fig_save_name, tick_sizes=20, tick_skip=1, k_range=None)
        # plt.show()
        if ret_ax:
            return sample_labels, ax
        else:
            return sample_labels

    def scale_samples(self, samples, fine_ctrl):
        samples_cpy = copy.deepcopy(samples)
        samples_cpy[0, :] = samples_cpy[0, :] / (self.x_max - self.x_min)
        samples_cpy[1, :] = samples_cpy[1, :] / (self.y_max - self.y_min)
        samples_cpy = samples_cpy*fine_ctrl
        return samples_cpy

    def gen_ds_samples(self, box2numsamples_dict, viz=True, fig_save_name='test'):
        samples = np.zeros((2, 0))
        surf_id = np.zeros(0)
        for box_id in box2numsamples_dict.keys():
            num_samples = box2numsamples_dict[box_id]
            redundant_sample_mult_factor = 25
            temp_samples = self.boxes[box_id].get_random_vectors(num_samples=num_samples*redundant_sample_mult_factor)
            """
            NOTE: With the way code is set up, boxes are allowed to overlap. However, at the intersection of two boxes,
            the box which has the higher surface index gets priority. This code block just checks if the samples generated
            for a particular box will actually be assigned a different surface value than self.box2surfid[box_id] and if it does,
            they get pruned. This is also the reason for the redundant sample mult factor param above. Redundant samples 
            above the user-specified limit will be pruned by random choice.
            """
            base_ids = np.ones(num_samples*redundant_sample_mult_factor)*self.box2surfid[box_id]
            _, temp_sample_labels = self.assign_samples2surfidx(temp_samples)
            temp_samples = temp_samples[:, np.where(temp_sample_labels == base_ids)[0]]
            # Pruning any remaining redundant samples to adhere to the user-specified sample limits.
            # print(box_id)
            # print(temp_samples.shape)
            temp_samples = temp_samples[:, np.random.choice(a=np.arange(temp_samples.shape[-1]),
                                                            size=num_samples,
                                                            replace=False)]
            samples = np.hstack([samples, temp_samples])
            surf_id = np.hstack([surf_id, np.ones(temp_samples.shape[1])*box_id])
        if viz:
            self.viz_samples(samples, fig_save_name=fig_save_name)
        return samples

    def get_mode_at_points(self, samples, enforce_satisfaction=False, skip_overlap_check=False):
        surf2satisfied_dict, sample_labels = self.assign_samples2surfidx(samples, skip_overlap_check=skip_overlap_check)
        mode_deltas = np.zeros((self.num_surf, sample_labels.shape[0]))
        for i in range(self.num_surf):
            if surf2satisfied_dict.get(i, None) is None:
                continue
            mode_deltas[i, :] = surf2satisfied_dict[i]
        if enforce_satisfaction:
            valid_idxs = np.where(np.sum(mode_deltas, axis=0) == 1)[0]
            mode_deltas = mode_deltas[:, valid_idxs]
            sample_labels = sample_labels[valid_idxs]
            return mode_deltas, sample_labels, samples[:, valid_idxs]
        else:
            return mode_deltas, sample_labels
