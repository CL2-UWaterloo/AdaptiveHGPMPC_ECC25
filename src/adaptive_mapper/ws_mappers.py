import copy
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import ListedColormap
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

from sys_dyn.nlsys_utils import test_quad_2d_track
from sys_dyn.problem_setups import quad_2d_sys_1d_inp_res
from ds_utils import GP_DS
from .utils import Traj_DS, Mapping_DS
from adaptive_mapper.utils import gen_dl_from_samples_n_env

from common.plotting_utils import save_fig
from consts import FIG_SAVE_BOOL


def gen_traindl_from_mapping_ds(ws_samples, posteriors):
    train_x = ws_samples.T
    train_dataloader = gen_dl_from_samples_n_env(box_env_inst=None, samples=train_x, ret_ds=False, labels=posteriors)
    return train_dataloader
