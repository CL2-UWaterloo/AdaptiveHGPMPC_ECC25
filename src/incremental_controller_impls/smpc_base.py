import torch
import numpy as np
import casadi as cs
import scipy
import matplotlib.pyplot as plt

from common.box_constraint_utils import box_constraint
from common.plotting_utils import generate_fine_grid, AxisAdjuster, save_fig
from ds_utils.utils import GP_DS
from common.data_save_utils import save_to_data_dir
from consts import FIG_SAVE_BOOL


def setup_terminal_costs(A, B, Q, R):
    Q_lqr = Q
    R_lqr = R
    from scipy.linalg import solve_discrete_are
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    btp = np.dot(B.T, P)
    K = -np.dot(np.linalg.inv(R + np.dot(btp, B)), np.dot(btp, A))
    return K, P


def get_inv_cdf(n_i, satisfaction_prob, boole_deno_override=False):
    # \overline{p} from the paper
    if not boole_deno_override:
        p_bar_i = 1 - (1 / n_i - (satisfaction_prob + 1) / (2 * n_i))
    else:
        p_bar_i = 1 - ((1 - satisfaction_prob) / 2)
    # \phi^-1(\overline{p})
    inverse_cdf_i = scipy.stats.norm.ppf(p_bar_i)
    return inverse_cdf_i


class delta_matrix_mult(cs.Callback):
    def __init__(self, name, n_in_rows, n_in_cols, num_regions, N, opts={}, delta_tol=0, test_softplus=False):
        cs.Callback.__init__(self)
        self.n_in_rows = n_in_rows
        # number of columns in a single matrix corresponding to 1 region. Total columns when they're hstacked will be
        # num_regions * n_in_cols_single
        self.n_in_cols = n_in_cols
        self.num_regions = num_regions
        self.N = N
        self.delta_tol = delta_tol
        self.test_softplus = test_softplus
        self.sharpness_param = 75
        self.construct(name, opts)

    def get_n_in(self):
        # 1 for delta array followed by num_regions number of matrices
        return 1+self.num_regions

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.num_regions, 1)
        else:
            return cs.Sparsity.dense(self.n_in_rows, self.n_in_cols)

    def get_sparsity_out(self, i):
        return cs.Sparsity.dense(self.n_in_rows, self.n_in_cols)

    def eval(self, arg):
        delta_k, matrix_arr = arg[0], arg[1:]
        matrix_summation = cs.DM.zeros(self.n_in_rows, self.n_in_cols)
        for region_idx in range(len(matrix_arr)):
            if not self.test_softplus:
                delta_vec = delta_k[region_idx, 0]
            else:
                delta_vec = np.log(1+np.exp(self.sharpness_param*delta_k[region_idx, 0]))/self.sharpness_param
            matrix_summation += matrix_arr[region_idx] * (delta_vec + self.delta_tol)
        return [matrix_summation]


class hybrid_res_covar(delta_matrix_mult):
    def __init__(self, name, n_d, num_regions, N, opts={}, delta_tol=0, test_softplus=False):
        super().__init__(name, n_d, n_d, num_regions, N, opts, delta_tol, test_softplus)


class computeSigmaglobal_meaneq(cs.Callback):
    # flag for taylor approx
    def __init__(self, name, feedback_mat, residual_dim, opts={}):
        cs.Callback.__init__(self)
        self.K = feedback_mat
        self.n_u, self.n_x = self.K.shape
        self.n_d = residual_dim
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 3
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.n_x, self.n_x)
        elif i == 1:
            return cs.Sparsity.dense(self.n_u, self.n_u)
        elif i == 2:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    # This method is needed to specify Sigma's output shape matrix (as seen from casadi's callback.py example). Without it,
    # the symbolic output Sigma is treated as (1, 1) instead of (mat_dim, mat_dim)
    def get_sparsity_out(self, i):
        # Forward sensitivity
        mat_dim = self.n_x+self.n_u+self.n_d
        return cs.Sparsity.dense(mat_dim, mat_dim)

    def eval(self, arg):
        Sigma_x, Sigma_u, Sigma_d = arg[0], arg[1], arg[2]
        # print(Sigma_x.shape, Sigma_u.shape, Sigma_d.shape)
        assert Sigma_d.shape == (self.n_d, self.n_d), "Shape of Sigma_d must match with n_d value specified when creating instance"
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L310
        Sigma_xu = Sigma_x @ self.K.T
        # Sigma_xu = Sigma_x @ self.K
        # Sigma_zd is specific to mean equivalence
        Sigma_xd = np.zeros((self.n_x, self.n_d))
        Sigma_ud = np.zeros((self.n_u, self.n_d))

        Sigma_z = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu),
                             cs.horzcat(Sigma_xu.T, Sigma_u)
                            )

        Sigma = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu, Sigma_xd),
                           cs.horzcat(Sigma_xu.T, Sigma_u, Sigma_ud),
                           cs.horzcat(Sigma_xd.T, Sigma_ud.T, Sigma_d))

        return [Sigma]


class computeSigma_nofeedback(cs.Callback):
    # flag for taylor approx
    def __init__(self, name, n_x, n_u, n_d, opts={}):
        cs.Callback.__init__(self)
        self.n_x, self.n_u = n_x, n_u
        self.n_d = n_d
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cs.Sparsity.dense(self.n_x, self.n_x)
        elif i == 1:
            return cs.Sparsity.dense(self.n_d, self.n_d)

    # This method is needed to specify Sigma's output shape matrix (as seen from casadi's callback.py example). Without it,
    # the symbolic output Sigma is treated as (1, 1) instead of (mat_dim, mat_dim)
    def get_sparsity_out(self, i):
        # Forward sensitivity
        mat_dim = self.n_x+self.n_u+self.n_d
        return cs.Sparsity.dense(mat_dim, mat_dim)

    def eval(self, arg):
        Sigma_x, Sigma_d = arg[0], arg[1]
        # print(Sigma_x.shape, Sigma_u.shape, Sigma_d.shape)
        assert Sigma_d.shape == (self.n_d, self.n_d), "Shape of Sigma_d must match with n_d value specified when creating instance"
        # https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/controllers/mpc/gp_mpc.py#L310

        Sigma_xu = np.zeros((self.n_x, self.n_u))
        Sigma_u = np.zeros((self.n_u, self.n_u))
        Sigma_xd = np.zeros((self.n_x, self.n_d))
        Sigma_ud = np.zeros((self.n_u, self.n_d))

        Sigma = cs.vertcat(cs.horzcat(Sigma_x, Sigma_xu, Sigma_xd),
                           cs.horzcat(Sigma_xu.T, Sigma_u, Sigma_ud),
                           cs.horzcat(Sigma_xd.T, Sigma_ud.T, Sigma_d))

        return [Sigma]


def planar_region_gen_and_viz(viz=True, s_start_limit=np.array([[-2, -2]]).T, s_end_limit=np.array([[2, 2]]).T,
                              x0_delim=-0.5, x1_delim=0.5, ax=None):
    # print("delimiters", x0_delim, x1_delim)
    # r1 spans full x1 but x0 \in [-2, -0.5]
    r1_start, r1_end = np.array([[s_start_limit[0, :].item(), s_start_limit[1, :].item()]]).T,\
                       np.array([[x0_delim, s_end_limit[1, :].item()]]).T
    # r2 spans the remainder of x0 and x1 is limited to be 0.5 -> 2
    r2_start, r2_end = np.array([[x0_delim, x1_delim]]).T,\
                       np.array([[s_end_limit[0, :].item(), s_end_limit[1, :].item()]]).T
    # r3 also spans the remainder of x0 and now x1 too [-2, 0.5].
    r3_start, r3_end = np.array([[x0_delim, s_start_limit[1, :].item()]]).T,\
                       np.array([[s_end_limit[0, :].item(), x1_delim]]).T
    regions = [box_constraint(r1_start, r1_end), box_constraint(r2_start, r2_end), box_constraint(r3_start, r3_end)]
    if viz:
        visualize_regions(s_start_limit=s_start_limit, s_end_limit=s_end_limit, regions=regions, ax=ax)
    return regions


def visualize_regions(s_start_limit, s_end_limit, regions, ax=None):
    # Add values to generate samples that lie outside of the constraint set to test those too
    grid_check = generate_fine_grid(s_start_limit-1, s_end_limit+1, fineness_param=(45, 15), viz_grid=False)
    # print(grid_check.shape)
    mask = [[], [], []]
    for grid_vec_idx in range(grid_check.shape[-1]):
        grid_vec = grid_check[:, grid_vec_idx]
        for region_idx in range(len(regions)):
            test_constraint = regions[region_idx]
            mask[region_idx].append((test_constraint.sym_func(grid_vec) <= 0).all().item())
    passed_vecs = [0, 0, 0]
    colours = ['r', 'g', 'b']
    if ax is None:
        plt.figure()
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            plt.scatter(passed_vecs[i][0], passed_vecs[i][1], c=colours[i], alpha=0.3)
    else:
        for i in range(len(regions)):
            passed_vecs[i] = grid_check[:, mask[i]]
            ax.scatter(passed_vecs[i][0], passed_vecs[i][1], c=colours[i], alpha=0.1)
    # print(grid_check)


def construct_config_opts(print_level, add_monitor, early_termination, hsl_solver,
                          test_no_lam_p, hessian_approximation=True, jac_approx=False):
    # Ref https://casadi.sourceforge.net/v1.9.0/api/html/dd/df1/classCasADi_1_1IpoptSolver.html
    opts = {"ipopt.print_level": print_level}
    if add_monitor:
        # opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
        opts["monitor"] = ["nlp_g"]
    if test_no_lam_p:
        opts["calc_lam_p"] = False
    if hessian_approximation:
        opts["ipopt.hessian_approximation"] = "limited-memory"
    if jac_approx:
        opts["ipopt.jacobian_approximation"] = "finite-difference-values"
    if hsl_solver:
        opts["ipopt.linear_solver"] = "ma27"
    acceptable_dual_inf_tol = 1e4
    acceptable_compl_inf_tol = 1e-1
    acceptable_iter = 5
    acceptable_constr_viol_tol = 5*1e-2
    acceptable_tol = 1e5
    max_iter = 200

    if early_termination:
        additional_opts = {"ipopt.acceptable_tol": acceptable_tol, "ipopt.acceptable_constr_viol_tol": acceptable_constr_viol_tol,
                           "ipopt.acceptable_dual_inf_tol": acceptable_dual_inf_tol, "ipopt.acceptable_iter": acceptable_iter,
                           "ipopt.acceptable_compl_inf_tol": acceptable_compl_inf_tol, "ipopt.max_iter": max_iter}
        opts.update(additional_opts)
    return opts


def construct_config_opts_minlp(print_level, add_monitor=False, early_termination=True, hsl_solver=False,
                                test_no_lam_p=True, hessian_approximation=True, jac_approx=False):
    opts = {"bonmin.print_level": print_level, 'bonmin.file_solution': 'yes', 'bonmin.expect_infeasible_problem': 'no'}
    # opts.update({"bonmin.allowable_gap": -100, 'bonmin.allowable_fraction_gap': -0.1, 'bonmin.cutoff_decr': -10})
    opts.update({"bonmin.allowable_gap": 2})
    opts["bonmin.integer_tolerance"] = 1e-2

    if add_monitor:
        opts["monitor"] = ["nlp_g", "nlp_jac_g", "nlp_f", "nlp_grad_f"]
    if early_termination:
        opts["bonmin.solution_limit"] = 4
        opts.update({"bonmin.allowable_gap": 5})
    if hsl_solver:
        opts['bonmin.linear_solver'] = 'ma27'
    if jac_approx:
        opts["bonmin.jacobian_approximation"] = "finite-difference-values"
    if test_no_lam_p:
        opts["calc_lam_p"] = False
    if hessian_approximation:
        opts["bonmin.hessian_approximation"] = "limited-memory"


def fwdsim_w_pw_res(true_func_obj: GP_DS, ct_dyn_nom, dt, Bd, gp_input_mask, delta_input_mask, x_0, u_0, ret_residual=True,
                    no_noise=False, clip_to_region_fn=None, clip_state=False,
                    ret_collision_bool=False, integration_method='euler', dt_dyn_nom=None, multi_var_n_inp=False, **kwargs):
    x_0, u_0 = np.array(x_0, ndmin=2).reshape(-1, 1), np.array(u_0, ndmin=2).reshape(-1, 1)
    # x_0_delta = clip_to_region_fn(x_0) if clip_to_region_fn is not None else x_0
    x_0_delta = clip_to_region_fn(x_0) if clip_to_region_fn is not None else x_0  # Clipping to account for slight optimization errors ex: even -2e-8 prevents region idx assignment

    delta_input_0 = torch.from_numpy((delta_input_mask @ np.vstack([x_0_delta, u_0])).astype(np.float32))
    # print(delta_input_0)
    # print([str(region) for region in true_func_obj.regions])
    delta_dict = true_func_obj._generate_regionspec_mask(input_arr=delta_input_0, delta_var_inp=True)

    if multi_var_n_inp:
        gp_input_masks = kwargs.get("gp_input_masks")
        num_gp_vars = len(gp_input_masks)
        gp_inputs_0 = [torch.from_numpy((gp_input_masks[var_idx] @ np.vstack([x_0_delta, u_0])).astype(np.float32)) for var_idx in range(num_gp_vars)]
        """
        Note: The way the GP_DS is set up, if for example we had 3 regions and 2 variables to model it could theoretically generate
        the 6 residual values (3x2) in a single call of the generate_outputs method. The problem is that it expects the same gp_input i.e.
        the same gp_input_mask to be applied to the joint state-input vector. This is not the case for an autonomous vehicle using a 
        kinematic model for example. As a result, we call num_gp_vars number of times and isolate the correct values at each instance. 
        """
        sampled_residual = np.zeros((true_func_obj.output_dims, 1))  # n_d x 1
        for var_idx in range(num_gp_vars):
            sampled_residual[var_idx] = true_func_obj.generate_outputs(input_arr=gp_inputs_0[var_idx], no_noise=no_noise, return_op=True, noise_verbose=False,
                                                                       mask_dict_override=delta_dict, ret_noise_sample=False)
    else:
        gp_input_0 = torch.from_numpy((gp_input_mask @ np.vstack([x_0_delta, u_0])).astype(np.float32))
        sampled_residual = true_func_obj.generate_outputs(input_arr=gp_input_0, no_noise=no_noise, return_op=True, noise_verbose=False,
                                                          mask_dict_override=delta_dict, ret_noise_sample=False)

    # print([str(region) for region in true_func_obj.regions])
    # print(x_0, sampled_residual, delta_dict)
    # # print(ct_dyn_nom(x=x_0, u=u_0)['f'])
    # # print(dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy()))
    # print(x_0 + dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy()))

    # Sampled residual = region_res_mean + w_k where w_k is sampled for region stochasticity function (assumed here to be gaussian).
    if integration_method == 'euler':
        sampled_ns = x_0 + dt*(ct_dyn_nom(x=x_0, u=u_0)['f'] + Bd @ sampled_residual.numpy())
    else:
        sampled_ns = dt_dyn_nom(x=x_0, u=u_0)['xf'] + (dt * Bd @ sampled_residual.numpy())
    sampled_ns_clipped = clip_to_region_fn(sampled_ns) if clip_state is not False else sampled_ns
    if np.isclose(sampled_ns_clipped, sampled_ns, atol=1e-4).all():
        collision = False
    else:
        # print("Collision detected. %s, %s" % (sampled_ns_clipped, sampled_ns))
        collision = True
    # print(x_0)
    # print(dt*(ct_dyn_nom(x=x_0, u=u_0)['f']))
    # print(dt*Bd @ sampled_residual.numpy())
    # print(sampled_ns)
    # print("Sampled residual: %s, dt*sampled res: %s" % (sampled_residual.numpy(), (dt*Bd @ sampled_residual.numpy())))

    ret = [sampled_ns_clipped]
    if ret_residual:
        ret = ret + [sampled_residual]
    if ret_collision_bool:
        ret = ret + [collision]
    return ret


class OptiDebugger:
    def __init__(self, controller_inst):
        self.controller_inst = controller_inst

    def get_vals_from_opti_debug(self, var_name):
        assert var_name in self.controller_inst.opti_dict.keys(), "Controller's opti_dict has no key: %s . Add it to the dictionary within the O.L. setups" % var_name
        if type(self.controller_inst.opti_dict[var_name]) in [list, tuple]:
            return [self.controller_inst.opti.debug.value(var_k) for var_k in self.controller_inst.opti_dict[var_name]]
        else:
            return self.controller_inst.opti.debug.value(self.controller_inst.opti_dict[var_name])


def plot_OL_opt_soln(debugger_inst: OptiDebugger, state_plot_idxs, ax=None, waypoints_to_track=None,
                     ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='upper center'):
    mu_x_ol = np.array(debugger_inst.get_vals_from_opti_debug('mu_x'), ndmin=2)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(mu_x_ol[state_plot_idxs[0], :].squeeze(), mu_x_ol[state_plot_idxs[1], :].squeeze(), color="r", marker='x',
            linestyle='dashed', linewidth=2, markersize=12, label='OL output')
    if waypoints_to_track is not None:
        ax.plot(waypoints_to_track[0, :], waypoints_to_track[1, :], color='g', marker='o', linestyle='solid',
                label='Trajectory to track')
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    ax.legend(loc=legend_loc)


def plot_CL_opt_soln(waypoints_to_track, data_dict_cl, ret_mu_x_cl, state_plot_idxs, plot_ol_traj=False, axes=None,
                     ax_xlim=(-1, 8), ax_ylim=(-1, 8), legend_loc='lower right', regions=None, plot_idxs=(1, 3),
                     ignore_legend=False, itn_num=1, colour='r', label='CL Trajectory', box_env_inst=None, fig_name='test',
                     ret_ax=False, fig=None, title_text=''):
    mu_x_cl = [data_dict_ol['mu_x'] for data_dict_ol in data_dict_cl]

    if plot_ol_traj:
        if axes is None:
            fig, axes = plt.subplots(2, 1)
        else:
            assert len(axes) == 2, "axes must be a list of length 2 if plot_ol_traj=True so OL traj can be plotted on separate graph"
    else:
        if axes is None:
            fig, axes = plt.subplots(1, 1)
            axes = [axes]
        else:
            assert len(axes) == 1, "axes must be a list of length 1 if plot_ol_traj=False"


    for ax in axes:
        if waypoints_to_track is not None:
            ax.plot(waypoints_to_track[0, :], waypoints_to_track[1, :], color='cyan',
                    label='Trajectory to track', marker='o', linewidth=3, markersize=10)
        if not ignore_legend:
            ax.legend(loc='lower right')
        if box_env_inst is not None:
            pass
            # box_env_inst.visualize_env(alpha=0.1, ax=ax)
        elif regions is not None:
            colours = ['r', 'g', 'b']
            for region_idx, region in enumerate(regions):
                # if type(region.plot_idxs) is not list:
                #     region.plot_idxs = list(plot_idxs)
                region.plot_constraint_set(ax, colour=colours[region_idx], alpha=0.6)

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
    mu_x_z_cl = [(mu_x_ol[state_plot_idxs[0], 0], mu_x_ol[state_plot_idxs[1], 0]) for mu_x_ol in mu_x_cl]
    x_cl_traj_to_plot, z_cl_traj_to_plot = np.array([mu_x_k[0] for mu_x_k in mu_x_z_cl]).squeeze(), np.array([mu_x_k[1] for mu_x_k in mu_x_z_cl])
    axes[0].plot(x_cl_traj_to_plot,
                 z_cl_traj_to_plot,
                 color=colour, marker='o', linewidth=3, markersize=10, label=label)
    AxisAdjuster(labelsize=25).adjust_ax_object(ax=axes[0],
                                                title_text=title_text,
                                                xlabel_text='x-coord (m)',
                                                ylabel_text='z-coord (m)', set_equal=False, skip_legend=False, legend_loc="upper center")


    if plot_ol_traj:
        colours = ['cyan', 'r', 'g', 'b'] * (len(mu_x_cl) // 4 + 1)
        for timestep, mu_x_ol in enumerate(mu_x_cl):
            axes[1].plot(mu_x_ol[state_plot_idxs[0], :].squeeze(), mu_x_ol[state_plot_idxs[1], :].squeeze(), color=colours[timestep], marker='x',
                         linestyle='dashed', linewidth=3, markersize=10)

    for ax in axes:
        if not ignore_legend:
            ax.legend(loc=legend_loc, fontsize=15)

    # if fig is not None:
    data_dict = {"ref_traj": waypoints_to_track, "cl_traj": np.hstack([x_cl_traj_to_plot.reshape(1, -1), z_cl_traj_to_plot.reshape(1, -1)]), "box_env_inst": box_env_inst}
    save_to_data_dir(data_dict, file_name=fig_name)
    # save_to_data_dir(fig, file_name=fig_name)
    if FIG_SAVE_BOOL:
        save_fig(axes=axes, fig_name=fig_name, tick_sizes=15, tick_skip=1, k_range=None)

    # save_fig(axes=axes, fig_name="cl_traj_mapping_"+str(itn_num), tick_sizes=16)

    if ret_mu_x_cl:
        if ret_ax:
            return mu_x_cl, axes
        return mu_x_cl


def calc_cl_cost(mu_x_cl, mu_u_cl, x_desired, u_desired, Q, R):
    x_cost, u_cost = 0, 0
    for sim_step in range(len(mu_x_cl)-1):
        x_des_dev, u_des_dev = (mu_x_cl[sim_step] - x_desired[:, sim_step]), (mu_u_cl[sim_step] - u_desired[:, sim_step])
        x_cost += x_des_dev.T @ Q @ x_des_dev
        u_cost += u_des_dev.T @ R @ u_des_dev
    x_final_dev = (mu_x_cl[-1] - x_desired[:, len(mu_x_cl)-1])
    x_cost += x_final_dev.T @ Q @ x_final_dev
    return x_cost + u_cost
