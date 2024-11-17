from collections import OrderedDict

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common.plotting_utils import AxisAdjuster
from adaptive_mapper.kde_utils import test_vanilla_OKDE
from adaptive_mapper.kde_utils import interpolate
from common.data_save_utils import save_to_data_dir, read_data, update_data
from consts import FIG_SAVE_BOOL


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, softmax=False,
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            if softmax:
                softmax_layer = nn.Softmax(dim=1)
                self.net.append(softmax_layer)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

        self.init_training_modules()

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

    def init_training_modules(self):
        self.optim = torch.optim.Adam(lr=1e-4, params=self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def siren_training_loop(self, train_dl, total_steps=250, num_mse=1, num_ce=3, steps_til_summary=10, save_final=False,
                            test_ds=None, test_ds_fine_ctrl=None):
        self.train()

        for step in range(total_steps // (num_mse+num_ce)):
            model_input, gp_posterior_labels = next(iter(train_dl))
            for i in range(num_mse+num_ce):
                model_output, _ = self(model_input)
                softmaxd_op = torch.softmax(model_output.squeeze(), dim=1)
                if i < num_mse:
                    loss = torch.sum(((softmaxd_op.squeeze() - gp_posterior_labels)**2).squeeze(), 1).mean()
                else:
                    loss = self.criterion(model_output, gp_posterior_labels).mean()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (step % steps_til_summary) == 0:
                print("Step %d, Total loss %0.6f" % (step * (num_mse+num_ce), loss))


def init_siren_inst(omega_param=0.75):
    img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                      hidden_layers=3, outermost_linear=True, softmax=False,
                      first_omega_0=omega_param, hidden_omega_0=omega_param)
    img_siren.double()
    return img_siren


def one_hot_to_categorical(hard_labels, num_modes):
    categorical_arr = [-1, ] * hard_labels.shape[-1]
    for mode_idx in range(num_modes):
        mode_labels = hard_labels[mode_idx, :]
        for sample_idx in list(np.where(mode_labels == 1)[0]):
            categorical_arr[sample_idx] = mode_idx
    return categorical_arr


class SirenPredictor:
    def __init__(self, siren_inst: Siren):
        self.siren_inst = siren_inst
        self.siren_inst.eval()
        self.num_modes = self.siren_inst.out_features

    def get_siren_predictions(self, model_input):
        model_input = torch.from_numpy(model_input)
        model_output, coords = self.siren_inst(model_input.T)
        softmaxd_op = torch.softmax(model_output.squeeze(), dim=1)
        return model_output.T, softmaxd_op.T, coords.T

    def gen_hard_labels_from_soft(self, soft_labels):
        # Assumes soft_labels of shape (num_modes, num_samples)
        hard_labels = torch.zeros_like(soft_labels)
        hard_labels[torch.argmax(soft_labels, dim=0), torch.arange(soft_labels.shape[-1])] = 1
        return hard_labels

    def convert_hard_labels_to_categorical_arr(self, hard_labels):
        categorical_arr = one_hot_to_categorical(hard_labels, self.num_modes)
        return categorical_arr

    def get_pred_acc(self, true_labels, pred_labels):
        assert (pred_labels.shape == true_labels.shape)
        columnwise_equality = np.all(true_labels == pred_labels, axis=0)
        correct_count = len(np.where(columnwise_equality == 1)[0])
        acc = correct_count / true_labels.shape[1]
        return acc

    def gen_pred_outputs(self, test_X):
        with torch.no_grad():
            test_outputs, softmaxd_op, _ = self.get_siren_predictions(test_X)
            # test_loss = self.siren_inst.criterion(test_outputs, test_y)
        hard_test_ops = self.gen_hard_labels_from_soft(soft_labels=softmaxd_op)
        categorical_arr = self.convert_hard_labels_to_categorical_arr(hard_labels=hard_test_ops)
        return test_outputs, softmaxd_op, hard_test_ops, categorical_arr

    def test_model_preds(self, test_X, test_y=None, colour_overrides=None):
        self.siren_inst.eval()  # Set the model to evaluation mode
        test_outputs, softmaxd_op, hard_test_ops, categorical_arr = self.gen_pred_outputs(test_X=test_X)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(test_X[0, :], test_X[1, :], c=np.array(categorical_arr).squeeze(),
                   cmap=ListedColormap(['r', 'g', 'b'] if colour_overrides is None else colour_overrides),
                   vmin=0, vmax=2)
        if test_y is not None:
            acc = self.get_pred_acc(test_y, hard_test_ops.numpy())

        if FIG_SAVE_BOOL:
            pass
            # save_fig(axes=[ax[1]], fig_name=fig_save_base+"hard_"+str(step*(num_mse+num_ce)), tick_sizes=20, tick_skip=1, k_range=None)
        self.siren_inst.train()
        return ax

    def compute_okde_to_alpha_map(self, old_samples, kde_bw=0.2, min_cutoff_sigma=1, max_cutoff_sigma=1, max_cutoff_num=2, alpha_min=0.1, alpha_max=0.4):
        okde_inst, okde_min_cutoff, okde_max_cutoff = test_vanilla_OKDE(old_samples, bandwidth=kde_bw,
                                                                        min_cutoff_sigma=min_cutoff_sigma, max_cutoff_sigma=max_cutoff_sigma, max_cutoff_num=max_cutoff_num,
                                                                        viz=False, ret_min_max_cutoffs=True)
        OKDE_to_alpha_map = lambda x: interpolate(x, okde_min_cutoff, okde_max_cutoff, alpha_min, alpha_max)
        return okde_inst, OKDE_to_alpha_map

    def compute_posteriors(self, new_samples, old_samples, likelihoods_new, kde_bw=0.2, min_cutoff_sigma=1, max_cutoff_sigma=1, max_cutoff_num=2, alpha_min=0.1, alpha_max=0.4):
        okde_inst, alpha_map_fn = self.compute_okde_to_alpha_map(old_samples, kde_bw=kde_bw, min_cutoff_sigma=min_cutoff_sigma, max_cutoff_sigma=max_cutoff_sigma,
                                                                 max_cutoff_num=max_cutoff_num, alpha_min=alpha_min, alpha_max=alpha_max)
        alpha_map_vals = alpha_map_fn(okde_inst.estimate_density(new_samples.T)).T
        _, softmaxd_op, _, _ = self.gen_pred_outputs(test_X=new_samples)
        unnormd_posteriors_new = (likelihoods_new ** alpha_map_vals) * softmaxd_op.detach().numpy()
        samplewise_norm_const = np.linalg.norm(unnormd_posteriors_new, ord=1, axis=0).reshape(1, -1)
        labels_new = unnormd_posteriors_new / samplewise_norm_const
        return labels_new

    def get_pred_mse(self, true_labels, soft_pred_labels):
        mse_loss = np.mean((soft_pred_labels - true_labels) ** 2, axis=0)
        total_mse_loss = np.mean(mse_loss)
        return total_mse_loss

    def test_model_preds_img(self, test_X, test_y, test_ds_fine_ctrl, viz_samples_on_test_pred=True, fig_save_name=None, samples_to_viz=None,
                             acc_save_file_name='test'):
        self.siren_inst.eval()  # Set the model to evaluation mode
        test_outputs, softmaxd_op, hard_test_ops, categorical_arr = self.gen_pred_outputs(test_X=test_X)
        self.siren_inst.train()
        acc = self.get_pred_acc(test_y, hard_test_ops.numpy())
        mse = self.get_pred_mse(test_y, softmaxd_op.numpy())
        acc_prior = read_data(file_name=acc_save_file_name)
        acc_prior.append(acc)
        update_data(new_data=acc_prior, file_name=acc_save_file_name)
        mse_pred_prior = read_data(file_name=acc_save_file_name+"_mse")
        mse_pred_prior.append(mse)
        update_data(new_data=mse_pred_prior, file_name=acc_save_file_name+"_mse")
        print("PRED ACC %s; PRED MSE %s" % (acc, mse))
        fine_ctrl = test_ds_fine_ctrl

        arrs_to_viz = [softmaxd_op, hard_test_ops]
        for i in range(2):
            fig, ax = viz_quad_map_img(arr_to_viz=arrs_to_viz[i].T, fine_ctrl=fine_ctrl, samples_to_plot=None if not viz_samples_on_test_pred else samples_to_viz)
            AxisAdjuster(labelsize=25).adjust_ax_object(ax=ax,
                                                        title_text="%s label visualization across the workspace (Acc: %s)" % ("Soft" if i == 0 else "Hard", acc),
                                                        xlabel_text='x-coord (m)',
                                                        ylabel_text='z-coord (m)', set_equal=False, skip_legend=True)
            if FIG_SAVE_BOOL and fig_save_name is not None:
                save_name = fig_save_name+"%s"%("_soft" if i==0 else "_hard")
                save_to_data_dir(fig, file_name=save_name)
                plt.savefig(save_name+'.svg', format='svg', dpi=300, bbox_inches='tight')


def viz_quad_map_img(arr_to_viz, fine_ctrl, samples_to_plot=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    im2viz = arr_to_viz.view(fine_ctrl, fine_ctrl, 3).numpy()
    ax.imshow(im2viz, origin="lower", extent=[0, 4, 0, 4])
    if samples_to_plot is not None:
        ax.scatter(samples_to_plot[0, :], samples_to_plot[1, :], color="orange", s=60)
    return fig, ax
