import dill as pkl
import os
import consts


def save_acc_n_loss(data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                    file_name="acc_n_loss_save", extension='.pkl'):
    if os.path.exists(base_path+file_name+extension):
        skip_read = False
    else:
        skip_read = True
    if not skip_read:
        with open(base_path+file_name+extension, 'rb') as f:
            base_data = pkl.load(f)
    else:
        base_data = {'acc': [], 'loss': []}
    with open(base_path+file_name+extension, 'wb') as f:
        base_data['acc'].append(data['acc'])
        base_data['loss'].append(data['loss'])
        pkl.dump(base_data, f)


def read_acc_n_loss(base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                    file_name="acc_n_loss_save", extension='.pkl', print_data=True):
    with open(base_path+file_name+extension, 'rb') as f:
        data = pkl.load(f)
    if print_data:
        print(data)
    return data


def update_file(update_data, file_name):
    try:
        with open(file_name+".pkl", "rb") as file:
            curr_data = pkl.load(file)
    except (EOFError, FileNotFoundError):
        curr_data = []

    with open(file_name+".pkl", "wb") as file:
        if type(update_data) is dict:
            curr_data.append(update_data)
        elif type(update_data) is list:
            curr_data.extend(update_data)
        pkl.dump(curr_data, file)


def load_file(file_name):
    try:
        with open(file_name+".pkl", "rb") as file:
            curr_data = pkl.load(file)
    except (EOFError, FileNotFoundError):
        print("File empty/not found. Returning empty list")
        curr_data = []
    return curr_data


def save_to_data_dir(data, file_name):
    cwd = os.getcwd()
    os.chdir(consts.data_dir)

    with open(file_name+".pkl", "wb") as pklfile:
        pkl.dump(data, pklfile)
    os.chdir(cwd)


def save_data(data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
              file_name="gpmpc_d_runs", extension='.pkl', update_data=False):
    file_exists = False
    if os.path.exists(base_path+file_name+extension):
        file_exists = True
        if not update_data:
            file_name = file_name + '_1'
        else:
            with open(base_path+file_name+extension, 'rb') as f:
                old_data = pkl.load(f)
    with open(base_path+file_name+extension, 'wb') as f:
        if update_data and file_exists:
            if len(old_data) > len(data):
                data = old_data + data
        pkl.dump(data, f)


def read_data(base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
              file_name="gpmpc_d_runs", extension='.pkl'):
    # assert os.path.exists(base_path+file_name+extension), "File to read desired trajectory to track does not exist"
    try:
        with open(base_path+file_name+extension, 'rb') as f:
            data = pkl.load(f)
    except (EOFError, FileNotFoundError) as e:
        with open(base_path+file_name+extension, 'wb') as f:
            pkl.dump([], f)
        return []
    return data


def update_data(new_data, base_path="C:\\Users\\l8souza\\PycharmProjects\\GPMPC_HM\\src\\data_dir\\",
                file_name="gpmpc_d_runs", extension='.pkl'):
    assert os.path.exists(base_path+file_name+extension), "File to read desired trajectory to track does not exist"
    with open(base_path+file_name+extension, 'wb') as f:
        pkl.dump(new_data, f)
