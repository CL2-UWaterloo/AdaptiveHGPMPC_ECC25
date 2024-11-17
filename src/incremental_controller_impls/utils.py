import numpy as np
from incremental_controller_impls.smpc_base import setup_terminal_costs
from adaptive_mapper.utils import Box_Environment


def linearize_for_terminal(sym_dyn_model, x_desired, n_u, Q, R):
    partial_der_calc = sym_dyn_model.df_func(x=x_desired, u=np.zeros((n_u, 1)))
    A_xf, B_xf = partial_der_calc['dfdx'], partial_der_calc['dfdu']
    _, P_val = setup_terminal_costs(A_xf, B_xf, Q, R)
    return P_val


def test_box_env_3_box_3_surf(viz=True):
    num_boxes = 3
    num_surf = 3
    # higher box idx gets priority over lower.
    lb_arr = [np.array([[0, -0.05]]).T, np.array([[1, 2]]).T, np.array([[1, -0.05]]).T]
    ub_arr = [np.array([[1, 4]]).T, np.array([[4, 4]]).T, np.array([[4, 2]]).T]
    # lb_arr = [np.array([[1, 2]]).T, np.array([[0, -0.05]]).T, np.array([[1, -0.05]]).T]
    # ub_arr = [np.array([[4, 4]]).T, np.array([[1, 4]]).T, np.array([[4, 2]]).T]
    x_min, x_max, y_min, y_max = 0, 4, 0, 4
    test_box_env = Box_Environment(num_boxes=num_boxes, num_surf=num_surf,
                                   lb_arr=lb_arr, ub_arr=ub_arr,
                                   box2surfid={2: 2, 1: 1, 0: 0},
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    # print(test_box_env)
    if viz:
        test_box_env.visualize_env()

    return test_box_env


def adaptive_box_env_3_box_3_surf(viz=True):
    num_boxes = 3
    num_surf = 3
    # higher box idx gets priority over lower.
    lb_arr = [np.array([[0, -0.05]]).T, np.array([[1, 2]]).T, np.array([[1, -0.05]]).T]
    ub_arr = [np.array([[1, 4]]).T, np.array([[4, 4]]).T, np.array([[4, 2]]).T]
    # lb_arr = [np.array([[1, 2]]).T, np.array([[0, -0.05]]).T, np.array([[1, -0.05]]).T]
    # ub_arr = [np.array([[4, 4]]).T, np.array([[1, 4]]).T, np.array([[4, 2]]).T]
    x_min, x_max, y_min, y_max = 0, 4, 0, 4
    test_box_env = Box_Environment(num_boxes=num_boxes, num_surf=num_surf,
                                   lb_arr=lb_arr, ub_arr=ub_arr,
                                   box2surfid={2: 2, 1: 0, 0: 0},
                                   x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    if viz:
        test_box_env.visualize_env()

    return test_box_env
