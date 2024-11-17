import copy
import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from gp_models.utils import pw_gp_viz
from ds_utils import GP_DS
from common.box_constraint_utils import box_constraint
from common.plotting_utils import save_fig
from consts import FIG_SAVE_BOOL


class Empty_Traj_DS:
    def __init__(self, pw_gp_inp_dim, measured_res_dim, delta_dim, batch_size=None):
        self.pw_gp_inp_dim = pw_gp_inp_dim
        self.res_output_dim = measured_res_dim
        self.delta_dim = delta_dim
        self.reset_vars()
        self.create_torch_vars()

        self.batch_size = batch_size

    def reset_vars(self):
        self.pw_gp_inp, self.measured_res, self.delta_control_vec = [np.empty((dim, 0)) for dim in (self.pw_gp_inp_dim, self.res_output_dim, self.delta_dim)]

    def create_torch_vars(self):
        self.pw_gp_inp_torch, self.measured_res_torch, self.delta_control_vec_torch = [torch.from_numpy(var_value) for var_value in (self.pw_gp_inp, self.measured_res, self.delta_control_vec)]

    def size_check(self, pw_gp_inp, measured_res, delta_control_vec, batch_size_check=False):
        assert pw_gp_inp.shape[0] == self.pw_gp_inp_dim, "pw_gp_inp_new must have shape (pw_gp_inp_dim, num_samples)"
        assert measured_res.shape[0] == self.res_output_dim, "measured_res_new must have shape (res_output_dim, num_samples)"
        assert delta_control_vec.shape[0] == self.delta_dim, "delta_control_vec_new must have shape (delta_dim, num_samples)"
        if batch_size_check:
            if self.batch_size is not None:
                assert pw_gp_inp.shape[1] == measured_res.shape[1] == delta_control_vec.shape[1] == self.batch_size, "All inputs must have the same number of samples, i.e. batch_size"

    def update_ds(self, pw_gp_inp_new, measured_res_new, delta_control_vec_new, batch_size_check=False):
        self.size_check(pw_gp_inp_new, measured_res_new, delta_control_vec_new, batch_size_check=batch_size_check)

        self.pw_gp_inp = np.hstack((self.pw_gp_inp, pw_gp_inp_new))
        self.measured_res = np.hstack((self.measured_res, measured_res_new))
        self.delta_control_vec = np.hstack((self.delta_control_vec, delta_control_vec_new))
        self.create_torch_vars()

    def shuffle(self):
        perm = np.random.permutation(self.pw_gp_inp.shape[1])
        self.pw_gp_inp = self.pw_gp_inp[:, perm]
        self.measured_res = self.measured_res[:, perm]
        self.delta_control_vec = self.delta_control_vec[:, perm]
        self.create_torch_vars()


class Traj_DS(Empty_Traj_DS):
    def __init__(self, pw_gp_inp, measured_res, delta_control_vec, batch_size=None):
        pw_gp_inp_dim, measured_res_dim, delta_dim = pw_gp_inp.shape[0], measured_res.shape[0], delta_control_vec.shape[0]
        super().__init__(pw_gp_inp_dim, measured_res_dim, delta_dim, batch_size)

        self.update_ds(pw_gp_inp, measured_res, delta_control_vec)


class Mapping_DS:
    def __init__(self, gp_ds_inst: GP_DS, pw_gp_wrapped, num_test_samples=5000, delta_dim=2, compare_with_gt=False,
                 batch_size=8, traj_ds_override=None):
        self.gp_ds_inst = gp_ds_inst
        self.pw_gp_wrapped = pw_gp_wrapped
        self.num_test_samples = num_test_samples
        self.compare_with_gt = compare_with_gt

        self.pw_gp_inp_dim, self.res_output_dim = gp_ds_inst.input_dims, gp_ds_inst.output_dims
        self.delta_dim = delta_dim
        self.batch_size = batch_size

        # All samples are stored in the traj_ds_inst
        if traj_ds_override is None:
            self.traj_ds_inst = Empty_Traj_DS(pw_gp_inp_dim=self.pw_gp_inp_dim, measured_res_dim=self.res_output_dim, delta_dim=self.delta_dim)
        else:
            self.traj_ds_inst = traj_ds_override
        # Only partial traj inst for soft label ds gen. Contains subtrajectories of size batch_size
        self.partial_traj_ds = Empty_Traj_DS(pw_gp_inp_dim=self.pw_gp_inp_dim, measured_res_dim=self.res_output_dim, delta_dim=self.delta_dim, batch_size=self.batch_size)

        # Static until called again.
        self.gen_test_ds()
        self.init_nn_ds()

        if traj_ds_override is not None:
            self.update_softlabelds_from_traj(partial=False)

    def init_nn_ds(self):
        train_data_placeholder = torch.empty((self.delta_dim, 0))
        train_label_placeholder = torch.empty((self.gp_ds_inst.num_surf, 0))
        gt_labels_arr = torch.empty((self.gp_ds_inst.num_surf, 0))
        gt_check_arr = torch.empty((1, 0))
        nn_ds = {"train_labels": train_label_placeholder, "train_data": train_data_placeholder, "train_gt_labels": gt_labels_arr,
                 "test_data": self.test_data, "test_labels": self.test_mask, "gt_check_arr": gt_check_arr}
        retain_keys = ["test_data", "test_labels"]
        nn_ds_batch = {key: (value if key in retain_keys else None) for key, value in nn_ds.items()}
        self.nn_ds, self.nn_ds_batch = nn_ds, nn_ds_batch

    def update_ds(self, pw_gp_inp_new, measured_res_new, delta_control_vec_new):
        self.partial_traj_ds.reset_vars()
        # print(pw_gp_inp_new.shape)
        self.partial_traj_ds.update_ds(pw_gp_inp_new, measured_res_new, delta_control_vec_new)  # Only partial traj inst for soft label ds gen. Prevents needing to generate soft labels
                                                                                                # for full traj ds every time since earlier traj's have already had their labels generated previously.
        self.update_softlabelds_from_traj(partial=True)
        # print("generated soft labels for partial traj ds")
        self.traj_ds_inst.update_ds(pw_gp_inp_new, measured_res_new, delta_control_vec_new)

    def gen_test_ds(self):
        _, self.test_mask = self.gp_ds_inst.generate_fine_grid(fineness_param=None, num_samples=self.num_test_samples, with_mask=True)
        self.test_data = self.gp_ds_inst.generate_random_delta_var_assignments(self.test_mask, delta_dim=self.delta_dim)

    def train_label_gt_check(self, train_data, train_labels):
        gt_labels_dict = self.gp_ds_inst._generate_regionspec_mask(input_arr=train_data.numpy(), delta_var_inp=True)
        gt_labels_arr = stack_mask_dict(train_labels, self.gp_ds_inst.num_surf, gt_labels_dict)
        assert gt_labels_arr.shape[-1] == self.gp_ds_inst.num_surf, "gt_labels_arr must have shape (num_samples, num_regions)"

        hard_labels = gen_hard_labels_from_soft(soft_labels=train_labels, num_samples=train_data.shape[-1])
        assert hard_labels.shape[-1] == self.gp_ds_inst.num_surf, "hard_labels must have shape (num_samples, num_regions)"
        gt_check_arr = np.all((hard_labels == gt_labels_arr).numpy(), axis=1)
        accuracy = (hard_labels == gt_labels_arr).sum().item() / (hard_labels.shape[0] * self.gp_ds_inst.num_surf)
        print("Accuracy of GP predictions on train_ds by argmax: %s" % accuracy)
        ce_loss = cross_entropy_loss_torch(soft_labels=train_labels, one_hot_labels=gt_labels_arr)
        print("Cross entropy loss of GP predictions on train_ds: %s" % (ce_loss/gt_labels_arr.shape[0]))
        data = {"acc": accuracy, "loss": ce_loss/gt_labels_arr.shape[0]}
        print(data)
        # save_acc_n_loss(data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
        #                 file_name="acc_n_loss_save_explore_scale_point5_no_red", extension='.pkl')
        return gt_labels_arr, gt_check_arr

    def choose_ds(self, partial):
        return self.partial_traj_ds if partial else self.traj_ds_inst

    def update_softlabelds_from_traj(self, partial=True):
        ds_inst = self.choose_ds(partial=partial)
        train_data = ds_inst.delta_control_vec_torch

        # train_labels are soft label vectors generated by gp library. gt_labels_arr are stacked ground truth labels for all regions
        train_labels = traj_soft_label_ds(ds_inst.measured_res_torch, self.pw_gp_wrapped, ds_inst.pw_gp_inp_torch)
        assert train_labels.shape[-1] == self.gp_ds_inst.num_surf, "train_labels must have shape (num_samples, num_regions)"

        gt_labels_arr = None
        gt_check_arr = None
        if self.compare_with_gt:
            gt_labels_arr, gt_check_arr = self.train_label_gt_check(train_data, train_labels)
            gt_check_arr = numpy.array(gt_check_arr, ndmin=2).reshape(1, -1)

        args = [train_data, train_labels.T]
        keys = ["train_data", "train_labels"]
        if self.compare_with_gt:
            args += [gt_labels_arr, gt_check_arr]
            keys += ["train_gt_labels", "gt_check_arr"]
        args = [(torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg) for arg in args]
        for key, arg in zip(keys, args):
            try:
                self.nn_ds[key] = torch.hstack((self.nn_ds[key], arg))
                self.nn_ds_batch[key] = arg
            except RuntimeError:
                assert RuntimeError(self.nn_ds[key].shape, arg.shape)

    def viz_sample_correctness(self, fineness_param, partial=True, itn_num=0):
        ds_inst = self.choose_ds(partial=partial)

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        pw_gp_viz(pw_gp_wrapped=self.pw_gp_wrapped, ds_ndim_test=self.gp_ds_inst, ax=ax, return_error_attrs=False,
                  fineness_param=fineness_param)
        x = ds_inst.pw_gp_inp_torch.squeeze()
        y = ds_inst.measured_res_torch.squeeze()
        correctness_labels = self.nn_ds["gt_check_arr"].squeeze()
        colour = ['r', 'b']
        labels = ['Incorrect by argmax', 'Correct by argmax']
        for correctness in np.unique(correctness_labels):
            ix = np.where(correctness_labels == correctness)
            ax.scatter(x.numpy()[ix], y.numpy()[ix], c=colour[int(correctness.item())], label=labels[int(correctness.item())])
        # ax.scatter(x.numpy(), y.numpy(), c=correctness_labels, cmap=ListedColormap(['r', 'b']))
        ax.legend(loc='upper right', fontsize=15)
        # save_fig(axes=[ax], fig_name="sample_correctness_viz_"+str(itn_num), tick_sizes=16)


def gen_soft_label_ds(gp_ds_inst: GP_DS, pw_gp_wrapped, gp_inp_x, delta_gt_dict):
    # Generate residual terms based on delta mask created above.
    obsvd_res = gp_ds_inst.generate_outputs(input_arr=gp_inp_x, no_noise=False, return_op=True, mask_dict_override=delta_gt_dict)
    # print(obsvd_res)
    # Generate soft labels based on likelihood that each (i/o) pair in (fine_grid, obsvd_res) was generated by a particular region's gp
    train_labels = pw_gp_wrapped.pred_likelihood(gp_inp_x, obsvd_res)

    gt_labels_arr = stack_mask_dict(train_labels, gp_ds_inst.num_surf, delta_gt_dict)

    return obsvd_res, train_labels, gt_labels_arr


def traj_soft_label_ds(obsvd_res, pw_gp_wrapped, gp_inp_x):
    # Generate soft labels based on likelihood that each (i/o) pair in (fine_grid, obsvd_res) was generated by a particular region's gp
    train_labels = pw_gp_wrapped.pred_likelihood(gp_inp_x, obsvd_res)
    return train_labels


def stack_mask_dict(arr_4_shape, num_regions, delta_gt_dict):
    gt_labels_arr = torch.zeros_like(arr_4_shape)
    for region_idx in range(num_regions):
        tens_gt_region_arr = torch.from_numpy(delta_gt_dict[region_idx]) if type(delta_gt_dict[region_idx]) is np.ndarray else delta_gt_dict[region_idx]
        gt_labels_arr[:, region_idx] = tens_gt_region_arr
    return gt_labels_arr


def gen_hard_labels_from_soft(soft_labels, num_samples):
    hard_labels = torch.zeros_like(soft_labels)
    # print(train_labels.shape, fine_grid.shape)
    hard_labels[torch.arange(num_samples), torch.argmax(soft_labels, dim=1)] = 1
    return hard_labels


def cross_entropy_loss_torch(soft_labels, one_hot_labels):
    # Convert numpy arrays to PyTorch tensors
    soft_labels = torch.tensor(soft_labels, dtype=torch.float32)
    one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.float32)

    # Avoid potential numerical instability by adding a small epsilon to the soft labels
    epsilon = 1e-10
    soft_labels = torch.clamp(soft_labels, epsilon, 1.0 - epsilon)

    # Compute the cross-entropy loss using PyTorch's functional API
    loss = -torch.sum(one_hot_labels * torch.log(soft_labels)) / soft_labels.size(1)

    return loss.item()  # Convert the loss back to a Python scalar


def mapping_ds_gen(gp_ds_inst: GP_DS, gt_test=False, num_train_samples=200, num_test_samples=5000, delta_dim=2,
                   pw_gp_wrapped=None, ds_override=None):
    """Creates a testing and training dataset from the given logfile.
    :param gt_test: If true, then in testing mode and the ground truth labels are used for mapping. If false, then labels are generated by wrapped gps.
    :param gp_ds_inst the GP_DS instance containing data sample attributes
    :param num_test_samples number of samples used to test the hilbert map classifier
    :param num_train_samples number of samples used in the dataset provided to the hilbert map. The trained classifier for the map
            is then used to assign class labels to each point in a fine grid of the workspace.
    :param delta_dim dimension of the delta variables that specify the regions in the workspace
    :return hm_ds containing gp inp samples, assignments to delta variables and also one hot vector encodings for each sample
            the one hot vector encodings can either be ground truth or gen'd from the wrapped gps (latter not implemented yet)
    """
    if gt_test is False:
        assert pw_gp_wrapped is not None, "pw_gp_wrapped must be provided if gt_test is False"
    if ds_override is not None:
        fine_grid, mask, delta_var_assgts = ds_override["gp_input_samples"], ds_override["train_labels"], ds_override["train_data"]
    else:
        if gp_ds_inst.region_gpinp_exclusive:
            # NOTE: Two ways to do this. Either generate fine grid over gp inputs and then assign to regions randomly or generate
            # the region random assignments first and then gp inputs. I think both generate the same result because of the random calls.

            # with_mask=True will generate a random mask (in the exclusive case) assigning gp input samples to regions
            fine_grid, mask = gp_ds_inst.generate_fine_grid(fineness_param=None, num_samples=num_train_samples, with_mask=True)
            # With this mask, we also need to randomly generate assignments to the delta variables using the region box constraints to use
            # during incremental training so we can start to assign class labels to specific points in the workspace using this incremental
            # approach.
            delta_var_assgts = gp_ds_inst.generate_random_delta_var_assignments(mask, delta_dim=delta_dim)
        else:
            raise NotImplementedError

    train_data = delta_var_assgts

    if gt_test:
        train_labels = mask.reshape(-1, gp_ds_inst.num_surf)  # ground truth labels (dict)
        gt_labels_arr = stack_mask_dict(train_labels, gp_ds_inst.num_surf, delta_gt_dict=mask)
        train_labels = gt_labels_arr
    else:
        # train_labels are soft label vectors generated by gp library. gt_labels_arr are stacked ground truth labels for all regions
        obsvd_res, train_labels, gt_labels_arr = gen_soft_label_ds(gp_ds_inst, pw_gp_wrapped, gp_inp_x=fine_grid, delta_gt_dict=mask)
        assert gt_labels_arr.shape[-1] == gp_ds_inst.num_surf, "gt_labels_arr must have shape (num_samples, num_regions)"


    test_likelihood = True
    if test_likelihood:
        hard_labels = gen_hard_labels_from_soft(soft_labels=train_labels, num_samples=fine_grid.shape[1])
        accuracy = (hard_labels == gt_labels_arr).sum().item() / (fine_grid.shape[-1] * gp_ds_inst.num_surf)
        print("Accuracy of GP predictions on train_ds by argmax: %s" % accuracy)
        ce_loss = cross_entropy_loss_torch(soft_labels=train_labels, one_hot_labels=gt_labels_arr)
        print("Cross entropy loss of GP predictions on train_ds: %s" % ce_loss)

        # print(train_labels)

    _, test_mask = gp_ds_inst.generate_fine_grid(fineness_param=None, num_samples=num_test_samples, with_mask=True)
    test_data = gp_ds_inst.generate_random_delta_var_assignments(test_mask, delta_dim=delta_dim)
    hm_ds = {"gp_input_samples": fine_grid,
             "train_labels": train_labels.T, "train_data": train_data,
             "test_data": test_data, "test_labels": test_mask}
    if test_likelihood and gt_test is False:
        hm_ds["hard_labels"] = hard_labels
        hm_ds["gt_labels"] = gt_labels_arr

    return hm_ds


def sort_by_angle(points, *args):
    for arg in args:
        assert arg.shape[1] == points.shape[1], "All arguments must have the same number of data points"
    angles = np.arctan2(points[1], points[0])
    sorted_indices = np.argsort(angles)
    return points[:, sorted_indices], sorted_indices


def convert_label_dict_to_categorical_array(train_y):
    """Converts the dictionary of labels to a single array.
    :param train_y the dictionary of labels
    :return the array of labels
    """

    train_y_merged = np.concatenate([train_y[idx] for idx in range(np.max(list(train_y.keys()))+1)], axis=0)
    # print(train_y_merged)

    # Convert the dictionary to a single array
    K = train_y[0].shape[1]
    categorical_labels = np.zeros((1, K), dtype=np.int)
    for i in range(K):
        categorical_labels[0, i] = np.argmax(train_y_merged[:, i]).item()
    return categorical_labels


def data_generator(train_x, train_y, batch_size=8):
    for i in range(0, train_x.shape[-1]//batch_size):
        yield train_x[:, i*batch_size:(i+1)*batch_size], train_y[:, i*batch_size:(i+1)*batch_size]


def get_mgrid(sidelen, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_testmap_tensor(sidelength, soft_vecs=False):
    from torchvision.transforms import Resize, Compose, ToTensor, Normalize
    size = sidelength
    ratio = size/512
    region_map = np.zeros((3, size, size)).reshape(-1, 3)

    x_range = np.linspace(0, 1, size)
    y_range = np.linspace(0, 1, size)

    # Create a grid of x[1] and x[2] values
    meshes = np.meshgrid(*[x_range, y_range], indexing='xy')

    # Flatten the meshgrid and stack the coordinates to create an array of size (K, n-dimensions)
    samples = np.vstack([m.flatten() for m in meshes])

    # print(samples[0, :].shape)

    # print((samples[0, :] < 128/size).shape)

    # Define the conditions
    region1_condition = (samples[0, :] < 128*ratio/size) & (samples[1, :] < 128*ratio/size)
    region2_condition = ((samples[0, :] > 128*ratio/size) & (samples[0, :] < 256*ratio/size) & (samples[1, :] < 256*ratio/size)) | \
                        ((samples[1, :] > 256*ratio/size) & (samples[0, :] > 256*ratio/size))
    print(region2_condition[int(256*ratio*size + 257)])
    # print(region2_condition[*150])

    region3_condition = ((samples[1, :] > 128*ratio/size) & (samples[0, :] < 128*ratio/size)) | \
                        ((samples[1, :] > 256*ratio/size) & (samples[0, :] > 128*ratio/size) & (samples[0, :] < 256*ratio/size)) | \
                        (samples[0, :] > 256*ratio/size) & (~region2_condition)

    if not soft_vecs:
        region_map[np.where(region1_condition)[0]] = np.array([1, 0, 0])
        region_map[np.where(region2_condition)[0]] = np.array([0, 1, 0])
        region_map[np.where(region3_condition)[0]] = np.array([0, 0, 1])
    else:
        num_region1 = np.where(region1_condition)[0].shape[0]
        num_region2 = np.where(region2_condition)[0].shape[0]
        num_region3 = np.where(region3_condition)[0].shape[0]
        region_num_samples = [num_region1, num_region2, num_region3]
        region_samples = []
        for i in range(3):
            soft_labels = generate_soft_labels_k(m=3, k=i, range_min=0.65, range_max=0.95, num_vectors=region_num_samples[i])
            region_samples.append(soft_labels)
        region_map = region_map.reshape(-1, 3).astype(np.float64)
        region_map[np.where(region1_condition)[0]] = region_samples[0]
        region_map[np.where(region2_condition)[0]] = region_samples[1]
        region_map[np.where(region3_condition)[0]] = region_samples[2]
        region_map = region_map.reshape(size, size, 3)
    region_map = (255*region_map.reshape(size, size, 3)).astype(np.uint8)

    # fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    # region_map_imgviz = torch.from_numpy(region_map)
    # ax.imshow(region_map_imgviz[:, :, [2]])
    # print(skimage.data.astronaut().max())
    # print(region_map.max())
    # print(region_map.min())

    img = Image.fromarray(region_map)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])
    img = transform(img)
    return img


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
                    colours=('r', 'g', 'b', 'cyan', 'black'), ret_ax=False):
        surf2satisfied_dict, sample_labels = self.assign_samples2surfidx(samples)
        # if alpha==1:
        #     print(surf2satisfied_dict)
        #     print(samples[0])
        #     print(samples[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.set_xlim(self.x_min, self.x_max)
        if scale_for_imshow:
            samples = self.scale_samples(samples, fine_ctrl)
        sc = ax.scatter(samples[0], samples[1],
                        c=sample_labels, cmap=ListedColormap(['r', 'g', 'b']), vmin=0, vmax=self.num_surf-1, alpha=alpha)
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


class RegionMap_DS(Dataset):
    def __init__(self, x, y, inp_dim=2, num_classes=3):
        assert y.shape[-1] == num_classes, "y must have %s number of columns but shape is: %s" % (num_classes, y.shape)
        assert x.shape[-1] == inp_dim, "x must have %s number of columns but shape is: %s" % (inp_dim, x.shape)
        self.x = x
        self.y = y

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.x, self.y


def gen_labels_from_samples_n_env(box_env_inst, samples):
    labels = np.vstack(box_env_inst.assign_samples2surf(samples=samples.T).values()).T.astype(float)
    return labels


def gen_dl_from_samples_n_env(box_env_inst, samples, ret_ds=False, labels=None):
    print("GEN DL FROM SAMPLES N ENV SAMPLES SHAPES: %s" % list(samples.shape))
    if labels is None:
        labels = gen_labels_from_samples_n_env(box_env_inst, samples)
    map_ds = RegionMap_DS(x=samples, y=labels)
    dataloader_inst = DataLoader(map_ds, batch_size=1, pin_memory=True, num_workers=0)
    if ret_ds:
        return dataloader_inst, map_ds
    else:
        return dataloader_inst


def gen_test_dl_from_env_inst(box_env_inst, fine_ctrl=100):
    test_x = box_env_inst.gen_test_samples(fine_ctrl=fine_ctrl).T
    test_dataloader = gen_dl_from_samples_n_env(box_env_inst=box_env_inst, samples=test_x)
    return test_dataloader


def gen_train_dl_from_env_inst(box_env_inst, box2numsamples_dict, scale_num=1, viz_samples=False, ret_ds=False, fig_save_name='test'):
    for box_idx in box2numsamples_dict.keys():
        box2numsamples_dict[box_idx] = box2numsamples_dict[box_idx] * scale_num
    train_x = box_env_inst.gen_ds_samples(box2numsamples_dict, viz=viz_samples, fig_save_name=fig_save_name).T
    train_dataloader = gen_dl_from_samples_n_env(box_env_inst=box_env_inst, samples=train_x, ret_ds=ret_ds)
    return train_dataloader


def get_testmap_tensor_initial(viz=True, fine_ctrl=100):
    num_boxes = 7
    num_surf = 3
    # NOTE: Clearly box 2 and box 4 overlap but the way the code is structured, box 4 (higher idx) gets priority over box 2 (lower idx) in the regions where they overlap
    lb_arr = [np.array([[0, 0]]).T, np.array([[0, 3.125]]).T, np.array([[1, 0]]).T, np.array([[2, 2]]).T, np.array([[0, 1]]).T, np.array([[1, 2]]).T, np.array([[2, 0]]).T]
    ub_arr = [np.array([[1, 1]]).T, np.array([[1.5625, 4]]).T, np.array([[2, 2]]).T, np.array([[4, 4]]).T, np.array([[1, 4]]).T, np.array([[2, 4]]).T, np.array([[4, 4]]).T]
    x_min, x_max, y_min, y_max = 0, 4, 0, 4
    test_box_env = Box_Environment(num_boxes=num_boxes, num_surf=num_surf,
                                   lb_arr=lb_arr, ub_arr=ub_arr,
                                   box2surfid={0: 2, 1: 2, 3: 2, 2: 1, 4: 0, 5: 0, 6: 0},
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if viz:
        test_box_env.visualize_env(fine_ctrl=fine_ctrl)

    return test_box_env


def get_testmap_tensor_adaptive(viz=True, fine_ctrl=100):
    num_boxes = 7
    num_surf = 3
    # NOTE: Clearly box 2 and box 4 overlap but the way the code is structured, box 4 (higher idx) gets priority over box 2 (lower idx) in the regions where they overlap
    lb_arr = [np.array([[0, 0]]).T, np.array([[0, 3.125]]).T, np.array([[1, 0]]).T, np.array([[2, 2]]).T, np.array([[0, 1]]).T, np.array([[1, 2]]).T, np.array([[2, 0]]).T]
    ub_arr = [np.array([[1, 1]]).T, np.array([[1.5625, 4]]).T, np.array([[2, 2]]).T, np.array([[4, 4]]).T, np.array([[1, 4]]).T, np.array([[2, 4]]).T, np.array([[4, 4]]).T]
    x_min, x_max, y_min, y_max = 0, 4, 0, 4
    test_box_env = Box_Environment(num_boxes=num_boxes, num_surf=num_surf,
                                   lb_arr=lb_arr, ub_arr=ub_arr,
                                   box2surfid={0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0},
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if viz:
        test_box_env.visualize_env(fine_ctrl=fine_ctrl)

    return test_box_env


def get_testmap_tensor_adaptive2(viz=True, fine_ctrl=100):
    num_boxes = 7
    num_surf = 3
    # NOTE: Clearly box 2 and box 4 overlap but the way the code is structured, box 4 (higher idx) gets priority over box 2 (lower idx) in the regions where they overlap
    lb_arr = [np.array([[0, 0]]).T, np.array([[0, 3.125]]).T, np.array([[1, 0]]).T, np.array([[2, 2]]).T, np.array([[0, 1]]).T, np.array([[1, 2]]).T, np.array([[2, 0]]).T]
    ub_arr = [np.array([[1, 1]]).T, np.array([[1.5625, 4]]).T, np.array([[2, 2]]).T, np.array([[4, 4]]).T, np.array([[1, 4]]).T, np.array([[2, 4]]).T, np.array([[4, 4]]).T]
    x_min, x_max, y_min, y_max = 0, 4, 0, 4
    test_box_env = Box_Environment(num_boxes=num_boxes, num_surf=num_surf,
                                   lb_arr=lb_arr, ub_arr=ub_arr,
                                   box2surfid={0: 2, 1: 2, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0},
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if viz:
        test_box_env.visualize_env(fine_ctrl=fine_ctrl)

    return test_box_env


def generate_numbers_with_sum(n, target_sum):
    # Generate n - 1 random integers between 1 and target_sum - 1
    numbers = [np.random.randint(1, target_sum - 1) for _ in range(n-1)]

    # Add 0 and target_sum as the start and end points
    numbers = [0] + numbers + [target_sum]

    # Sort the numbers in ascending order
    numbers.sort()

    # Calculate the differences between adjacent numbers
    differences = np.array([numbers[i+1] - numbers[i] for i in range(n)])
    np.random.shuffle(differences)

    return differences


def gen_test_grid(x_min, x_max, y_min, y_max, fine_control=75):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, fine_control), np.linspace(y_min, y_max, fine_control))
    return xx, yy


def generate_soft_labels_k(m, k, range_min, range_max, num_vectors):
    """
    m: Number of classes
    num_vectors: Number of soft label vectors to generate
    k: Index of the vector entry that should have the highest probability
    range_min, range_max: Range that the kth entry is generated between with the other entries utilizing the remaining space
    """
    vectors = np.zeros((num_vectors, m))

    for i in range(num_vectors):
        remaining_sum = 1.0
        vectors[i, k] = np.random.uniform(range_min, range_max)
        remaining_sum -= vectors[i, k]

        other_indices = np.delete(np.arange(m), k)
        np.random.shuffle(other_indices)

        for j in other_indices:
            vectors[i, j] = np.random.uniform(0, remaining_sum)
            remaining_sum -= vectors[i, j]

        vectors[i] /= np.sum(vectors[i])

    return vectors


def gen_traindl_from_mapping_ds(ws_samples, posteriors):
    train_x = ws_samples.T
    train_dataloader = gen_dl_from_samples_n_env(box_env_inst=None, samples=train_x, ret_ds=False, labels=posteriors)
    return train_dataloader

