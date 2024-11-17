from incremental_controller_impls.smpc_base import construct_config_opts
import incremental_controller_impls.smpc_base as controller_utils
from common.data_save_utils import read_data, update_data, save_to_data_dir
from common.plotting_utils import AxisAdjuster
from incremental_controller_impls.controller_classes import *
from sys_dyn.problem_setups import quad_2d_sys_1d_inp_res
from ds_utils import GP_DS
from incremental_controller_impls.utils import test_box_env_3_box_3_surf, adaptive_box_env_3_box_3_surf
from adaptive_mapper.ws_mappers import gen_traindl_from_mapping_ds
from adaptive_mapper.utils import Traj_DS, get_testmap_tensor_initial, get_testmap_tensor_adaptive2, gen_dl_from_samples_n_env
from adaptive_mapper.siren_map import SirenPredictor, init_siren_inst
from sys_dyn.nlsys_utils import test_quad_2d_track

from common.plotting_utils import save_fig
from consts import FIG_SAVE_BOOL


def trajds_setup_and_train(ds_inst_in: GP_DS, pw_gp_wrapped, num_test_samples=5000, delta_dim=2, compare_with_gt=True,
                           waypoints_arr=((1, 6), (2.5, 3), (0, 1)), init_traj_to_track=None,
                           simulation_length=95, num_discrete=95, verbose=False, N=30, velocity_override=5,
                           batch_size=8, expl_cost_fn=None, only_track=False,
                           ax=None, box_env_inst=None, sampling_time=50*10e-3,
                           siren_omega=5., test_dl_fine_ctrl=100, test_ds=None, total_steps=50, steps_til_summary=10, num_mse=1, num_ce=3,
                           viz_samples_on_test_pred=False, siren_fig_save_base='test', initial_traj_fig_save_name='test', acc_save_file_name='test', fig=None):
    """
    Runs a closed-loop simulation on a 2d quadrotor with nominal dynamics in the MPC but the true dynamics (with pw residual term)
    for closed-loop forward simulation.
    :return:
    """
    # sampling_time = 20 * 10e-3
    traj_ds_init: Traj_DS = test_quad_2d_track(test_jit=False, num_discrete=num_discrete, sim_steps=simulation_length,
                                               viz=True, verbose=verbose, N=N, velocity_override=velocity_override,
                                               ret_inputs=True, sampling_time=sampling_time,
                                               waypoints_arr=waypoints_arr, traj_to_track=init_traj_to_track,
                                               ax_xlim_override=[-0.05, box_env_inst.x_max], ax_ylim_override=[-0.05, box_env_inst.y_max], plot_wps=False,
                                               forward_sim="w_res_pw",
                                               forward_sim_kwargs={"true_func_obj": ds_inst_in,
                                                                   "Bd": quad_2d_sys_1d_inp_res()["Bd"],
                                                                   "gp_input_mask": quad_2d_sys_1d_inp_res()["gp_input_mask"],
                                                                   "delta_input_mask": quad_2d_sys_1d_inp_res()["delta_input_mask"]
                                                                   },
                                               expl_cost_fn=expl_cost_fn, ax=ax, box_env_2_viz=box_env_inst
                                               )
    AxisAdjuster(labelsize=25).adjust_ax_object(ax=ax,
                                                title_text="Run 1 (nominal controller)",
                                                xlabel_text='x-coord (m)',
                                                ylabel_text='z-coord (m)', set_equal=False, skip_legend=False, legend_loc="lower right")

    if FIG_SAVE_BOOL:
        x_z_coords = traj_ds_init.delta_control_vec
        data_dict = {"ref_traj": init_traj_to_track, "cl_traj": x_z_coords, "box_env_inst": box_env_inst}
        save_to_data_dir(data_dict, file_name=initial_traj_fig_save_name)
        save_fig(axes=[ax], fig_name=initial_traj_fig_save_name, tick_sizes=20, tick_skip=1, k_range=None)
    if only_track:
        return None, None

    mapping_ds_fn = lambda traj_ds_init: Mapping_DS(gp_ds_inst=ds_inst_in, pw_gp_wrapped=pw_gp_wrapped, num_test_samples=num_test_samples,
                                                    delta_dim=delta_dim, compare_with_gt=compare_with_gt, traj_ds_override=traj_ds_init, batch_size=batch_size)
    mapping_ds_inst = mapping_ds_fn(traj_ds_init)

    siren_inst = init_siren_inst(omega_param=siren_omega)
    train_dataloader = gen_traindl_from_mapping_ds(ws_samples=mapping_ds_inst.nn_ds["train_data"], posteriors=mapping_ds_inst.nn_ds["train_labels"].T)
    predictor_inst = train_siren_inst(siren_inst, train_dataloader, test_ds=test_ds, total_steps=total_steps, num_mse=num_mse, num_ce=num_ce, steps_til_summary=steps_til_summary,
                                      siren_fig_save_base=siren_fig_save_base, test_dl_fine_ctrl=test_dl_fine_ctrl, viz_samples_on_test_pred=viz_samples_on_test_pred,
                                      acc_save_file_name=acc_save_file_name)
    return mapping_ds_inst, siren_inst, predictor_inst


def train_siren_inst(siren_inst, train_dataloader, test_ds=None, total_steps=250, num_mse=0, num_ce=3, steps_til_summary=10,
                     siren_fig_save_base='test', test_dl_fine_ctrl=10, viz_samples_on_test_pred=False, acc_save_file_name='test'):
    siren_inst.siren_training_loop(train_dataloader, total_steps=total_steps, num_mse=num_mse, num_ce=num_ce, steps_til_summary=steps_til_summary)
    predictor_inst = SirenPredictor(siren_inst=siren_inst)
    if test_ds is not None:
        X_test, y_test = test_ds["X_test"], test_ds["y_test"]
        fig_save_name = siren_fig_save_base+str(total_steps*(num_mse+num_ce))
        predictor_inst.test_model_preds_img(test_X=X_test, test_y=y_test, test_ds_fine_ctrl=test_dl_fine_ctrl, viz_samples_on_test_pred=viz_samples_on_test_pred, fig_save_name=fig_save_name,
                                            samples_to_viz=next(iter(train_dataloader))[0].T.numpy(), acc_save_file_name=acc_save_file_name)
    return predictor_inst


def setup_test_dict_from_box(box_env_inst, fine_ctrl):
    test_x = box_env_inst.gen_test_samples(fine_ctrl=fine_ctrl)
    mode_deltas, _, _ = box_env_inst.get_mode_at_points(test_x, enforce_satisfaction=True)
    test_dict = {"X_test": test_x, "y_test": mode_deltas}
    return test_dict


def gpmpc_That_controller(init_traj_file, rep_traj_file, gp_fns, x_init, satisfaction_prob=0.95, velocity_override=0.7,
                          minimal_print=True, N=30, simulation_length=160, show_plot=True,
                          verbose=True, num_discrete=50, sampling_time=20 * 10e-3,
                          box_env_gen_fn=test_box_env_3_box_3_surf, new_box_env_gen_fn=adaptive_box_env_3_box_3_surf,
                          fwd_sim="w_pw_res", true_ds_inst: GP_DS=None,
                          collision_check=False, simulation_length_init=100,
                          online_N=None, use_prev_if_infeas=False, test_dl_fine_ctrl=100,
                          Q_override=None, R_override=None, Bd_override=None, num_mapping_updates=2,
                          siren_mapper=False, num_init_training_steps=50, num_rep_training_steps=20, steps_til_summary=25, num_mse=1, num_ce=3,
                          viz_samples_on_test_pred=False, kde_bw=0.5, kde_max_cutoff_num=2,
                          siren_omega=5., alpha_min=0.5, alpha_max=0.9, num_runs=20,
                          cost_save_file="cost_save_history", fixed_delta=False, run_num_offset=0, mode_switch_run_idx=2):
    box_env_inst = box_env_gen_fn(viz=False)
    sys_config_dict, inst_override = quad_2d_sys_1d_inp_res(velocity_limit_override=velocity_override, box_env_inst=box_env_inst,
                                                            sampling_time=sampling_time, ret_inst=True, region_viz=False)
    true_ds_inst.box_env_inst = sys_config_dict["box_env_inst"]
    if Q_override is not None:
        sys_config_dict["Q"] = Q_override
    if R_override is not None:
        sys_config_dict["R"] = R_override
    if Bd_override is not None:
        sys_config_dict["Bd"] = Bd_override
    lti = inst_override.lti
    integration_method = "euler"

    fn_config_dict = {"gp_fns": gp_fns, "horizon_length": N if online_N is None else online_N, 'piecewise': True,
                      'ignore_init_constr_check': True, "add_scaling": False,
                      "ignore_callback": False, "sampling_time": sampling_time,
                      "add_delta_constraints": False, "fwd_sim": fwd_sim,
                      "true_ds_inst": true_ds_inst, "Bd_fwd_sim": sys_config_dict["Bd"],
                      "infeas_debug_callback": False,
                      "satisfaction_prob": satisfaction_prob, "skip_shrinking": False,
                      "collision_check": collision_check, "boole_deno_override": False,
                      "lti": lti, "integration_method": integration_method, "use_prev_if_infeas": use_prev_if_infeas,
                      "siren_mapping": siren_mapper, "fixed_delta": fixed_delta, "predictor_type": "clean"}

    print_level = 0 if minimal_print else 5
    opts = construct_config_opts(print_level, add_monitor=False, early_termination=True, hsl_solver=False,
                                 test_no_lam_p=False, hessian_approximation=False, jac_approx=False)
    fn_config_dict["addn_solver_opts"] = opts

    configs = {}
    configs.update(sys_config_dict)
    configs.update(fn_config_dict)

    n_u = sys_config_dict["n_u"]
    # Nominal MPC solution for warmstarting.
    desired_traj_data = read_data(file_name=rep_traj_file)
    x_desired = desired_traj_data["x_desired"]
    u_desired = desired_traj_data["u_desired"]
    assert desired_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."
    u_desired = np.array(u_desired, ndmin=2).reshape((n_u, -1))
    tracking_matrix = sys_config_dict["tracking_matrix"]
    waypoints_to_track = tracking_matrix[:, :sys_config_dict["n_x"]] @ x_desired
    initial_info_dict = {"x_init": x_init, "u_warmstart": u_desired}

    init_traj_data = read_data(file_name=init_traj_file)
    x_desired_init = init_traj_data["x_desired"]
    assert init_traj_data["N"] >= online_N, "Desired trajectory has horizon length less than online horizon length."

    for run_num in range(num_runs):
        # Setup initial box environment
        box_env_inst: Box_Environment = box_env_gen_fn(viz=False)
        true_ds_inst.box_env_inst = box_env_inst
        # Will update regions in init method
        configs["box_env_inst"] = box_env_inst
        test_dict = setup_test_dict_from_box(box_env_inst=box_env_inst, fine_ctrl=test_dl_fine_ctrl)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        box_env_inst: Box_Environment = sys_config_dict["box_env_inst"]
        box_env_inst.visualize_env(fine_ctrl=100, ax=ax, alpha=0.2)
        ax.plot(x_desired_init[0, :], x_desired_init[2, :], color="cyan", marker='o',
                linestyle='solid', linewidth=3, markersize=7, markerfacecolor='cyan',
                label='Trajectory to track')

        save_prefix = "run_%s_" % (run_num+run_num_offset)
        acc_save_file_name = "%s_siren_training_argmax_acc_%s_%s" % ("fixed" if fixed_delta else "prop", alpha_min, alpha_max)
        mapping_ds_inst, siren_inst, predictor_inst = trajds_setup_and_train(ds_inst_in=true_ds_inst, pw_gp_wrapped=gp_fns,
                                                                             init_traj_to_track=x_desired_init, num_discrete=num_discrete, simulation_length=simulation_length_init,
                                                                             batch_size=simulation_length_init, box_env_inst=box_env_inst,
                                                                             ax=ax, sampling_time=sampling_time, total_steps=num_init_training_steps,
                                                                             viz_samples_on_test_pred=viz_samples_on_test_pred, siren_fig_save_base=save_prefix+'siren_init_train',
                                                                             initial_traj_fig_save_name=save_prefix+"initial_traj_cl", test_ds=test_dict,
                                                                             siren_omega=siren_omega, test_dl_fine_ctrl=test_dl_fine_ctrl,
                                                                             acc_save_file_name=acc_save_file_name, fig=fig)
        plt.show()

        mapping_ds_inst.batch_size = simulation_length
        fn_config_dict["That_predictor"] = predictor_inst
        fn_config_dict["initial_mapping_ds"] = mapping_ds_inst

        # Instantiate controller and setup optimization problem to be solved.
        configs = {}
        configs.update(sys_config_dict)
        configs.update(fn_config_dict)

        controller_inst = GPMPC_That(**configs)
        cl_costs = []

        for k in range(num_mapping_updates):
            # Setup mode shift
            if k+2 == mode_switch_run_idx:
                box_env_inst: Box_Environment = new_box_env_gen_fn(viz=False)
                true_ds_inst.box_env_inst = box_env_inst
                # Will update regions in init method
                configs["box_env_inst"] = box_env_inst
                test_dict = setup_test_dict_from_box(box_env_inst=box_env_inst, fine_ctrl=test_dl_fine_ctrl)

            data_dict_cl, sol_stats_cl = controller_inst.run_cl_opt(initial_info_dict, simulation_length=simulation_length,
                                                                    opt_verbose=verbose, x_desired=x_desired, u_desired=u_desired,
                                                                    infeas_debug_callback=False)

            print("CLOSED LOOP COST FOR ITERATION {}:".format(k))
            mu_x_cl = [data_dict_ol['mu_x'][:, 0] for data_dict_ol in data_dict_cl]
            mu_u_cl = [data_dict_ol['mu_u'][:, 0] for data_dict_ol in data_dict_cl]
            cl_cost = controller_utils.calc_cl_cost(mu_x_cl, mu_u_cl, x_desired, np.zeros((n_u, simulation_length)),
                                                    controller_inst.Q, controller_inst.R)
            print(cl_cost)
            cl_costs.append(cl_cost)
            file_save_suffix = '_itn%s' % (k+1)
            if show_plot:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                box_env_inst.visualize_env(fine_ctrl=100, alpha=0.2, ax=ax)
                fig_name = save_prefix+"cl_traj"+file_save_suffix
                mu_x_cl = controller_utils.plot_CL_opt_soln(waypoints_to_track=waypoints_to_track,
                                                            data_dict_cl=data_dict_cl,
                                                            ret_mu_x_cl=True,
                                                            state_plot_idxs=sys_config_dict["state_plot_idxs"],
                                                            plot_ol_traj=False, axes=[ax],
                                                            ax_xlim=(-0.05, box_env_inst.x_max), ax_ylim=(-0.05, box_env_inst.y_max), legend_loc='upper center',
                                                            box_env_inst=box_env_inst, itn_num=k+1, fig_name=fig_name, fig=fig, title_text="Run %s" % (k+2))

            new_batch_data = mapping_ds_inst.nn_ds_batch["train_data"].numpy()  # (2, 160)
            complete_sample_data = mapping_ds_inst.nn_ds["train_data"]
            old_sample_data = torch.flip(complete_sample_data, dims=(1,))[:, new_batch_data.shape[1]:].numpy()
            likelihoods_new = mapping_ds_inst.nn_ds_batch["train_labels"]
            posteriors = predictor_inst.compute_posteriors(new_batch_data, old_sample_data, likelihoods_new, kde_bw=kde_bw, min_cutoff_sigma=1, max_cutoff_sigma=1, max_cutoff_num=kde_max_cutoff_num,
                                                           alpha_min=alpha_min, alpha_max=alpha_max)

            train_dl_new = gen_dl_from_samples_n_env(box_env_inst=None, samples=new_batch_data.T, ret_ds=False, labels=posteriors.T)
            predictor_inst = train_siren_inst(siren_inst, train_dl_new, test_ds=test_dict, total_steps=num_rep_training_steps, num_mse=num_mse, num_ce=num_ce, steps_til_summary=steps_til_summary,
                                              siren_fig_save_base=save_prefix+'siren_retrained'+file_save_suffix,
                                              test_dl_fine_ctrl=test_dl_fine_ctrl, viz_samples_on_test_pred=viz_samples_on_test_pred, acc_save_file_name=acc_save_file_name)

            # Update mapping DS for use in next iteration.
            mapping_ds_inst.nn_ds["train_data"] = mapping_ds_inst.nn_ds_batch["train_data"]
            mapping_ds_inst.nn_ds["train_labels"] = mapping_ds_inst.nn_ds_batch["train_labels"]
            plt.show()
            controller_inst.That_predictor = predictor_inst

        cl_costs_prior = read_data(file_name=cost_save_file)
        cl_costs_prior.extend(cl_costs)
        update_data(new_data=cl_costs_prior, file_name=cost_save_file)


def alpha_tradeoff_ablation_test(pw_gp_wrapped, ds_inst_in, alpha_min_arr, num_runs_per_alpha=5, mode_switch_run_idx=4, num_mapping_updates=5,
                                 run_num_offset=52):
    Q_override = np.eye(6)
    Q_override[0, 0] = 20
    Q_override[2, 2] = 20
    R_override = np.eye(2) * 0.01

    for i, alpha_min in enumerate(alpha_min_arr):
        cost_save_file = "clcost_ablation_mapping_"+str(alpha_min)
        gpmpc_That_controller(init_traj_file="quad2d_dt50msN100disc100xz4_interiortrack_AIA_initial",
                              rep_traj_file="quad2d_dt50msN100disc160xz4_interiortrack_AIA_rep",
                              box_env_gen_fn=get_testmap_tensor_initial, new_box_env_gen_fn=get_testmap_tensor_adaptive2,
                              satisfaction_prob=0.99, gp_fns=pw_gp_wrapped, x_init=np.zeros((6, 1)), N=100, online_N=30,
                              num_discrete=160, velocity_override=5, fwd_sim="w_pw_res", true_ds_inst=ds_inst_in,
                              collision_check=True, verbose=False, sampling_time=50 * 1e-3, simulation_length_init=100, simulation_length=160,
                              num_mapping_updates=num_mapping_updates, Q_override=Q_override, R_override=R_override, siren_mapper=True,
                              num_init_training_steps=125, num_rep_training_steps=50,
                              viz_samples_on_test_pred=True, kde_bw=0.15, kde_max_cutoff_num=2, alpha_min=alpha_min, alpha_max=0.9,
                              num_runs=num_runs_per_alpha, siren_omega=5., mode_switch_run_idx=mode_switch_run_idx, cost_save_file=cost_save_file,
                              run_num_offset=run_num_offset)
        run_num_offset += num_runs_per_alpha+1
