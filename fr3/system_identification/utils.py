import os

from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml

from loguru import logger
from scipy.signal import savgol_filter
from six.moves import cPickle as pickle  # for performance

FR3_ROOT = Path(__file__).resolve().parents[1]

# ==============================================================================
# Customize wandb class for data logger
# ==============================================================================
import wandb


class WandbLogger(object):
    def __init__(self, project, exp_name, config):
        wandb.init(project=project, entity="junning", name=exp_name, config=config)

    # TODO: text_data_names: list of names for text data,
    # text_data: list of data
    # video_data_config: [filename of video_data, video_type, fps]
    # img_data_path: filename of image data
    def log_data(
        self,
        scalar_data_names,
        scalar_datas,
        video_data_config=None,
        img_data_path=None,
        table_data_config=None,
    ):
        data_dict = {}
        for name, data in zip(scalar_data_names, scalar_datas):
            data_dict.update({name: data})
        if video_data_config is not None:
            video_data_path, video_type, fps = video_data_config
            data_dict.update(
                {
                    "video": wandb.Video(
                        f"{video_data_path}", fps=fps, format=f"{video_type}"
                    )
                }
            )
        if img_data_path is not None:
            data_dict.update(
                {"image": wandb.Image(img_data_path)}
            )  # TODO: need testing
        if table_data_config is not None:
            table_keys, columns, datas = table_data_config
            for key, col, data in zip(table_keys, columns, datas):
                my_table = wandb.Table(columns=col, data=data)
                data_dict.update({key: my_table})
        wandb.log(data_dict)


# ==============================================================================
# Systematic tools: 1) load configurations; 2) return full path of a given file
# ==============================================================================
def _search_roots(path):
    raw_path = Path(path).expanduser()
    roots = []
    if raw_path.is_absolute():
        roots.append(raw_path)
    else:
        roots.append((Path.cwd() / raw_path).resolve())
        roots.append((FR3_ROOT / raw_path).resolve())
        if "robot_description" in raw_path.parts:
            roots.append(FR3_ROOT / "robot_description")

    unique_roots = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def find_path(name, path):
    for search_root in _search_roots(path):
        if not search_root.exists():
            continue
        for root, dirs, files in os.walk(search_root):
            if name in files:
                return os.path.join(root, name)
    searched = ", ".join(str(root) for root in _search_roots(path))
    raise Exception(f"Can't find {name} in directory {path}! Searched: {searched}")


def _find_robot_urdf(robot_name):
    urdf_file = f"{robot_name}.urdf"
    return find_path(urdf_file, "../robot_description")


def _load_pin_model_for_robot(robot_name):
    return pin.buildModelFromUrdf(_find_robot_urdf(robot_name))


def save_obj(obj_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(obj_, f)


def load_obj(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def save_yaml(filename, _dict):
    with open(filename, "w") as file:
        yaml.dump(_dict, file, default_flow_style=False)


def load_yaml(filename):
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def yml2urdf(yaml_file, urdf_template, urdf_file):
    import urdfpy

    _dict = load_yaml(yaml_file)
    robot = urdfpy.URDF.load(urdf_template)
    for i in range(2, len(robot.links) - 1):
        link_id = i - 1
        yml_data = _dict[f"link_{link_id}"]
        link = robot.links[i]
        ixx, iyy, izz, ixy, ixz, iyz = (
            yml_data["ixx"],
            yml_data["iyy"],
            yml_data["izz"],
            yml_data["ixy"],
            yml_data["ixz"],
            yml_data["iyz"],
        )
        link.inertial.inertia[0][0] = ixx
        link.inertial.inertia[0][1] = ixy
        link.inertial.inertia[1][0] = ixy
        link.inertial.inertia[0][2] = ixz
        link.inertial.inertia[2][0] = ixz
        link.inertial.inertia[1][1] = iyy
        link.inertial.inertia[2][2] = izz
        link.inertial.inertia[1][2] = iyz
        link.inertial.inertia[2][1] = iyz
        link.inertial.mass = yml_data["mass"]
        (
            link.inertial.origin[0][3],
            link.inertial.origin[1][3],
            link.inertial.origin[2][3],
        ) = (yml_data["xyz"][0], yml_data["xyz"][1], yml_data["xyz"][2])
    robot.save(urdf_file)


from copy import deepcopy


def merge_dict(dicts):
    _dict = deepcopy(dicts[0])
    if len(dicts) > 1:
        for key, item in dicts[0].items():
            for i in range(1, len(dicts)):
                _dict[key].update(dicts[i][key])
    return _dict


# ==============================================================================
# Load model parameters and predict toruqes
# ==============================================================================
def dynamics_prediction(models, q, dq, ddq, des_pos=None, des_vel=None, des_acc=None):
    if len(models) == 1:
        inertia_model = models[0]
        tau_inertia = inertia_model.predict(q, dq, ddq)
        tau = tau_inertia
    elif len(models) == 2:
        if des_vel is None:
            inertia_model, friction_model = models
            tau_inertia = inertia_model.predict(q, dq, ddq)
            tau_friction = friction_model.predict(q, dq, ddq)
            tau = tau_inertia + tau_friction
        else:
            inertia_model, friction_model = models
            tau_inertia = inertia_model.predict(q, dq, ddq)
            tau_friction = friction_model.predict(q, des_vel, ddq)
            tau = tau_inertia + tau_friction
    else:
        raise NotImplementedError
    return tau


# Load the inertia and friction model for torque prediction
def load_model_params(params, inertia_model, friction_model=None):
    total_param = inertia_model.njoints * 10
    dyn_param = params[:total_param]
    inertia_model.sync_param(dyn_param)
    if friction_model is not None:
        friction_param = params[total_param:]
        friction_model.sync_param(friction_param)


class Dataset(object):
    def __init__(self, dset_path, root_dir):
        root_dir = os.path.abspath(root_dir)
        dset_path = root_dir + "/" + dset_path
        self.dset = load_obj(dset_path)
        self.num_traj = len(self.dset["dset"])
        total_dset = self.dset["dset"]
        self.num_datapoints = np.sum([len(dset["t"]) for dset in total_dset])
        self.trainset_data, self.testset_data = self.get_trainset_testset()
        self.num_traindata, self.num_testdata = len(self.trainset_data[0]), len(
            self.testset_data[0]
        )
        self.dset_index = np.arange(self.num_traj)

    def get_train_data(self):
        (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            des_pos,
            des_vel,
            des_acc,
            tau_cmd,
        ) = self.trainset_data
        return (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            des_pos,
            des_vel,
            des_acc,
            tau_cmd,
        )

    def get_test_data(self):
        (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            des_pos,
            des_vel,
            des_acc,
            tau_cmd,
        ) = self.testset_data
        return (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            des_pos,
            des_vel,
            des_acc,
            tau_cmd,
        )

    def retrieve_dset_data_byindex(self, index):
        if index not in self.dset_index:
            raise f"Index is out of bound, we only have {self.num_traj} trajectories"
        else:
            dset = self.dset["dset"][index]
            return self.retrieve_dset_data(dset)

    def retrieve_dset_data(self, dset):
        t, q_f, dq_f, ddq_f, tau_f = (
            dset["t"],
            dset["q_f"],
            dset["dq_f"],
            dset["ddq_f"],
            dset["tau_f"],
        )
        q_m, dq_m, ddq_m, tau_m = (
            dset["q_m"],
            dset["dq_m"],
            dset["ddq_m"],
            dset["tau_m"],
        )
        des_pos, des_vel, des_acc = (
            dset["ref"][0]["desired_position"],
            dset["ref"][0]["desired_vel"],
            dset["ref"][0]["desired_acc"],
        )
        tau_cmd = dset["ref"][0]["tau_cmd"]
        return (
            t,
            q_f,
            dq_f,
            ddq_f,
            tau_f,
            q_m,
            dq_m,
            ddq_m,
            tau_m,
            des_pos,
            des_vel,
            des_acc,
            tau_cmd,
        )

    def merge_dsets(self, dsets):
        ts, qs_f, dqs_f, ddqs_f, taus_f, qs_m, dqs_m, ddqs_m, taus_m = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        des_poss, des_vels, des_accs = [], [], []
        tau_cmds = []
        for i in range(len(dsets)):
            dset = dsets[i]
            (
                t,
                q_f,
                dq_f,
                ddq_f,
                tau_f,
                q_m,
                dq_m,
                ddq_m,
                tau_m,
                des_pos,
                des_vel,
                des_acc,
                tau_cmd,
            ) = self.retrieve_dset_data(dset)
            ts.append(t)
            qs_f.append(q_f)
            dqs_f.append(dq_f)
            ddqs_f.append(ddq_f)
            taus_f.append(tau_f)
            qs_m.append(q_m)
            dqs_m.append(dq_m)
            ddqs_m.append(ddq_m)
            taus_m.append(tau_m)
            des_poss.append(des_pos)
            des_vels.append(des_vel)
            des_accs.append(des_acc)
            tau_cmds.append(tau_cmd)
        ts = np.vstack(ts)
        qs_f = np.vstack(qs_f)
        dqs_f = np.vstack(dqs_f)
        ddqs_f = np.vstack(ddqs_f)
        taus_f = np.vstack(taus_f)
        qs_m = np.vstack(qs_m)
        dqs_m = np.vstack(dqs_m)
        ddqs_m = np.vstack(ddqs_m)
        taus_m = np.vstack(taus_m)
        des_poss = np.vstack(des_poss)
        des_vels = np.vstack(des_vels)
        des_accs = np.vstack(des_accs)
        tau_cmds = np.vstack(tau_cmds)
        return [
            ts,
            qs_f,
            dqs_f,
            ddqs_f,
            taus_f,
            qs_m,
            dqs_m,
            ddqs_m,
            taus_m,
            des_poss,
            des_vels,
            des_accs,
            tau_cmds,
        ]

    def get_trainset_testset(self):
        dset = self.dset["dset"]
        train_set, test_set = self.split_train_test(dset)
        self.num_traj_training = len(train_set)
        self.num_traj_testing = len(test_set)
        trainset_data = self.merge_dsets(train_set)
        print("Generated training dataset")
        testset_data = self.merge_dsets(test_set)
        print("Generated testing dataset")
        return trainset_data, testset_data

    def split_train_test(self, dset):
        if len(dset) == 1:
            train_set = copy(dset)
            test_set = copy(dset)
            self.train_set_id = [0]
            self.test_set_id = [0]
        else:
            # by default all dataset except the last one would be
            # use as training data
            train_set_num = len(dset) // 3 * 2
            train_set = dset[:train_set_num]
            test_set = dset[train_set_num:]
            self.train_set_id = range(train_set_num)
            self.test_set_id = range(train_set_num, len(dset))
        return train_set, test_set


# ==============================================================================
# Robot configuration retriving with pinocchio
# ==============================================================================
def pin_joint_config(robot_name):
    # retrieve indexes for continus joints
    model = _load_pin_model_for_robot(robot_name)
    njoints = model.njoints - 1
    joint_configs = model.nqs[1:]
    idx_continuous_joint = []
    for i in range(njoints):
        if joint_configs[i] > 1:
            idx_continuous_joint.append(i)

    upper_joint_pos_limits = []
    lower_joint_pos_limits = []
    for joint_idx in range(njoints):
        if joint_idx in idx_continuous_joint:
            upper_pos_limit = np.pi * 4
            lower_pos_limit = -np.pi * 4
        else:
            upper_pos_limit = model.upperPositionLimit[joint_idx]
            lower_pos_limit = model.lowerPositionLimit[joint_idx]
        upper_joint_pos_limits.append(upper_pos_limit)
        lower_joint_pos_limits.append(lower_pos_limit)
    joint_vel_limits = model.velocityLimit
    return njoints, upper_joint_pos_limits, lower_joint_pos_limits, joint_vel_limits


def retrieve_robot_config(robot_name):
    (
        njoints,
        upper_joint_pos_limits,
        lower_joint_pos_limits,
        joint_vel_limits,
    ) = pin_joint_config(robot_name)
    init_pos = (
        np.array(upper_joint_pos_limits) + np.array(lower_joint_pos_limits)
    ) / 2.0
    init_vel = np.zeros(shape=len(upper_joint_pos_limits))
    config = {
        "njoints": njoints,
        "upper_joint_pos_limits": upper_joint_pos_limits,
        "lower_joint_pos_limits": lower_joint_pos_limits,
        "joint_vel_limits": joint_vel_limits,
        "init_pos": init_pos,
        "init_vel": init_vel,
    }
    return config


# ==============================================================================
# Visualization tools
# ==============================================================================
def draw_spectrum(x, fs):
    x = np.array(x)
    njoints = x.shape[1]
    fig, axes = plt.subplots(1)
    for i in range(njoints):
        xi = x[:, i]
        from scipy import signal

        f, Pxx_den = signal.periodogram(xi, fs)
        axes.semilogy(f, Pxx_den)
        # plt.xlim([0, 100])
        axes.set_xlabel("frequency [Hz]")
        axes.set_ylabel("PSD [V**2/Hz]")
        axes.set_title(f"Joint{i+1}")
    axes.legend([f"Joint{i+1}" for i in range(njoints)])


def vis_compare_seqs(t, seqs, legends, labels, mode=None):
    njoints = seqs[0].shape[1]

    fig, axes = plt.subplots(njoints, sharex=True)
    for i in range(njoints):
        for j in range(len(seqs)):
            seq = seqs[j][:, i]
            if njoints != 1:
                axes_i = axes[i]
            else:
                axes_i = axes
            if mode is None:
                axes_i.plot(t[j], seq)
            else:
                axes_i.plot(t[j], seq, ".")
        axes_i.set_ylabel(f"Joint {i+1}")
    plt.xlabel(labels[0])
    if njoints != 1:
        fig.align_ylabels(axes[:])
        legend = axes[-1].legend(legends)
    else:
        fig.align_ylabels(axes)
        legend = axes.legend(legends)
    # legend = axes[-1].legend(
    #     legends, loc="lower center", bbox_to_anchor=(0.11, -0.9), ncol=len(legends)
    # )
    legend.get_frame().set_edgecolor("b")
    legend.get_frame().set_linewidth(0.0)


def concanate_strings(lst_strings):
    output = ""
    for string in lst_strings:
        output += string
    return output


# ==============================================================================
# Linear algebra tools: 1) regularization; 2) Dimension reduction
# ==============================================================================
def tikhonov_regularization(A, lam=0.1):
    return A + lam * np.identity(A.shape[0])


def SVD_dim_reduction(A):
    # TODO: check why the identification fails
    u, s, v = np.linalg.svd(A, full_matrices=True)
    M, N = A.shape[0], A.shape[1]
    rank = np.linalg.matrix_rank(A)
    logger.info(f"Before dimension reduction, shape: {A.shape}, rank: {rank}")
    u_reduced = u[:, :rank]
    s_reduced = s[:rank]
    s_reduced_mat = np.diag(s_reduced)
    v_reduced = v[:rank, :rank]
    A_reduced = np.dot(u_reduced, np.dot(s_reduced_mat, v_reduced))

    logger.info(f"After dimension reduction, shape: {A_reduced.shape}, rank: {rank}")
    return A_reduced


def QR_dim_reduction_backup(A):
    """
    Perform Reduced QR decomposition, and remove the columns with small diagonal elements in R
    The resulted matrix has the same rank as the original matrix, the return include the reduced
    R matrix and the condition number of the reduced R matrix
    """
    q, r = scipy.linalg.qr(
        A, mode="economic"
    )  # scipy.linalg.qr is two times faster than np.linalg.qr
    M, N = A.shape[0], A.shape[1]
    rank = np.linalg.matrix_rank(r)

    diag_r = abs(np.diagonal(r))
    cols = np.arange(N)
    f = lambda x: diag_r[x]
    cols = sorted(cols, key=f)
    del_cols = cols[: N - rank]
    del_cols = sorted(del_cols)

    r_reduced = copy(r)
    r_reduced = np.delete(r_reduced, del_cols, 1)

    # compuate singular values of A
    cond = np.linalg.cond(r_reduced)
    return r_reduced, cond


def QR_dim_reduction(A):
    """
    Perform Reduced QR decomposition, and remove the columns with small diagonal elements in R.
    The resulted matrix has the same rank as the original matrix; returns the reduced R matrix
    and the condition number of the reduced R matrix.

    Avoids the double-SVD present in the backup version:
      - rank is estimated from the diagonal of R (no SVD needed for triangular matrices)
      - condition number is derived from the single SVD used to compute singular values
    """
    q, r = scipy.linalg.qr(A, mode="economic")
    N = A.shape[1]

    diag_r = np.abs(np.diagonal(r))
    tol = diag_r.max() * max(r.shape) * np.finfo(r.dtype).eps
    rank = int(np.sum(diag_r > tol))

    cols = np.argsort(diag_r)  # ascending: smallest diagonal first
    del_cols = np.sort(cols[: N - rank])

    r_reduced = np.delete(r, del_cols, axis=1)

    s = np.linalg.svd(r_reduced, compute_uv=False)
    cond = s[0] / s[-1]
    return r_reduced, cond, s


def feature2regressor(list_of_features, n_datapoints, njoints):
    """
    Transfer a list of features to regressor
        features: (n_datapoints, njoints)
        regressor: (n_datapoints*njoints, njoints)
    """
    check_features(list_of_features, n_datapoints, njoints)
    # expand each feature to the shape of regressor, (n_datapoints*njoints, njoints)
    list_of_features = [
        np.repeat(feature, njoints, axis=0) for feature in list_of_features
    ]

    # use identity matrix to filter out off-diagonal entries
    # identities_filter = []
    # for i in range(n_datapoints):
    #     identities_filter.append(np.identity(njoints))
    # identities_filter = np.vstack(identities_filter)
    identities_filter = np.tile(np.eye(njoints), (n_datapoints, 1))
    list_of_regressor = [feature * identities_filter for feature in list_of_features]
    regressor = np.hstack(list_of_regressor)
    return regressor


def check_features(list_of_features, n_datapoints, njoints):
    for i in range(len(list_of_features)):
        feature = list_of_features[i]
        assert (
            feature.shape[0] == n_datapoints and feature.shape[1] == njoints
        ), f"Feature{i} is not compatible with the given dataset."


# ==============================================================================
# Filtering and perturbations
# ==============================================================================
from scipy.signal import savgol_filter


def savitzy_filter(x, freq, window, poly_order=3, d_order=1, mode="nearest"):
    """
    Input:
        x: (nsamples, nD),
        freq: the frequency of the sampling data,
        window: moving window of the filter,
        poly_order: order for the polynomial
    Output:
        xd: (nsamples, nD)
    """
    xd = savgol_filter(x.T, window, poly_order, d_order, 1.0 / freq, mode=mode).T
    return xd


def perturbed_array(x, scale=1):
    noise = np.random.randn(x.shape[0], x.shape[1]) * scale
    x = x + noise
    return x


def check_jumps(x, shrehold=100):
    x_next = x[1:, :]
    x_prev = x[:-1, :]
    diff = abs(x_next - x_prev)
    diff = diff / abs(x_next)
    return np.any(diff >= shrehold)


# ==============================================================================
# Retrieve geometric center
# ==============================================================================
def retrieve_geo_fromCAD(robot_name="fr3"):
    model = _load_pin_model_for_robot(robot_name)
    njoints = model.njoints - 1  # the first joint is the universe joint
    system_inertia = model.inertias.tolist()[1 : 1 + njoints]  # skip the universe joint
    GoMs_lever = []
    masses_CAD = []
    inertias_CAD = []

    # Note that here inertia from CAD are at the CoM frame, not body frame
    for i in range(njoints):
        GoMs_lever.append(system_inertia[i].lever)
        masses_CAD.append(system_inertia[i].mass)
        inertias_CAD.append(system_inertia[i].inertia)
    return masses_CAD, GoMs_lever, inertias_CAD


def skew_symmetric(vec):
    if type(vec) == type([]):
        vec = np.array(vec)
    return np.array(
        [[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0]]
    )


def I(vec):
    ixx, ixy, iyy, ixz, iyz, izz = vec
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
    return np.log(x / (1 - x))


# ==================================================================================
# Tool functions for: 1) Pseudo inertia (4x4) to vector of dynamic parameters (1x10)
#                     2) Dynamic parameters (1x10) to pseudo inertia (4X4)
# ==================================================================================
def pinertiaToVec(J):
    m = J[3, 3]
    h = J[3, :3]
    # c = h / m
    Sigma = J[:3, :3]
    eye3 = np.eye(3)
    Ibar = np.trace(Sigma) * eye3 - Sigma
    Ixx, Iyy, Izz, Iyz, Ixz, Ixy = (
        Ibar[0][0],
        Ibar[1][1],
        Ibar[2][2],
        Ibar[2][1],
        Ibar[2][0],
        Ibar[1][0],
    )
    return np.array([m, h[0], h[1], h[2], Ixx, Ixy, Iyy, Ixz, Iyz, Izz])


def inertiaVecToinertiasCoM(pi_dyn_i):
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )

    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m

    Sc = skew_symmetric(c)
    Ic = Ibar - m * Sc @ Sc.T
    return m, c, Ic


def inertiaVecToPinertia(pi_dyn_i):
    # if the inertia in pi_dyn_i is on CoM, then
    # then J is on the CoM frame
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )

    # Ibar = np.array([
    #     [pi_dyn_i[4], pi_dyn_i[9], pi_dyn_i[8]],
    #     [pi_dyn_i[9], pi_dyn_i[5], pi_dyn_i[7]],
    #     [pi_dyn_i[8], pi_dyn_i[7], pi_dyn_i[6]]
    # ])

    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    eye3 = np.eye(3)
    Sigma = 1 / 2 * np.trace(Ibar) * eye3 - Ibar
    tmp1 = np.hstack([Sigma, np.reshape(h, (3, 1))])
    tmp2 = np.hstack([h, m])
    J = np.vstack([tmp1, tmp2])
    return J


def inertiaVecToIcQs(pi_dyn_i):
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )
    # Ibar = np.array([
    #     [pi_dyn_i[4], pi_dyn_i[9], pi_dyn_i[8]],
    #     [pi_dyn_i[9], pi_dyn_i[5], pi_dyn_i[7]],
    #     [pi_dyn_i[8], pi_dyn_i[7], pi_dyn_i[6]]
    # ])
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m
    Sc = skew_symmetric(c)
    Ic = Ibar - m * Sc @ Sc.T

    eye3 = np.eye(3)
    SigmaC = 1 / 2 * np.trace(Ic) * eye3 - Ic
    Qs = SigmaC / m
    return Ic, Qs


def inertiaVecToXsQs(pi_dyn_i):
    # Input here should be the center of mass estimated from the
    # CAD model
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )
    # Ibar = np.array([
    #     [pi_dyn_i[4], pi_dyn_i[9], pi_dyn_i[8]],
    #     [pi_dyn_i[9], pi_dyn_i[5], pi_dyn_i[7]],
    #     [pi_dyn_i[8], pi_dyn_i[7], pi_dyn_i[6]]
    # ])
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m
    Sc = skew_symmetric(c)
    Ic = Ibar - m * Sc @ Sc.T

    eye3 = np.eye(3)
    SigmaC = 1 / 2 * np.trace(Ic) * eye3 - Ic
    Qs = SigmaC / m
    return c, Qs


def inertiaVecToQ(pi_dyn_i):
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m
    c = c.reshape(-1, 1)

    _, Qs = inertiaVecToIcQs(pi_dyn_i)
    Qsinv = np.linalg.inv(Qs)
    QsinvTXs = Qsinv.T @ c
    XsTQsinvXs = c.T @ Qsinv @ c

    Q_col1 = np.hstack([-Qsinv, QsinvTXs])
    Q_col2 = np.hstack([QsinvTXs.T, 1 - XsTQsinvXs])
    Q = np.vstack([Q_col1, Q_col2])
    return Q


def param2dict(inertia_param, friction_param, njoints):
    inertia_param_dict = {}
    friction_param_dict = {}
    viscous_params = friction_param[:njoints].tolist()
    coulomb_params = friction_param[njoints:].tolist()
    for i in range(njoints):
        link = f"link_{i + 1}"
        v = viscous_params[i]
        c = coulomb_params[i]
        friction_coeff = {"damping": v, "friction": c}
        friction_param_dict[link] = friction_coeff
    for i in range(njoints):
        start_idx, end_idx = i * 10, (i + 1) * 10
        param = inertia_param[start_idx:end_idx]
        mass = float(param[0])
        xyz = [float(p) / mass for p in param[1:4]]
        l_inertia_body_frame = [float(p) for p in param[4:10]]
        I_inertia_body_frame = I(l_inertia_body_frame)
        skew_xyz = skew_symmetric(xyz)
        I_inertia_CoM_frame = I_inertia_body_frame - mass * (skew_xyz @ skew_xyz.T)
        ixx = float(I_inertia_CoM_frame[0][0])
        ixy = float(I_inertia_CoM_frame[0][1])
        iyy = float(I_inertia_CoM_frame[1][1])
        ixz = float(I_inertia_CoM_frame[0][2])
        iyz = float(I_inertia_CoM_frame[1][2])
        izz = float(I_inertia_CoM_frame[2][2])

        link = f"link_{i+1}"
        link_inertia = {
            "mass": mass,
            "xyz": xyz,
            "ixx": ixx,
            "ixy": ixy,
            "iyy": iyy,
            "ixz": ixz,
            "iyz": iyz,
            "izz": izz,
        }
        inertia_param_dict[link] = link_inertia

    param_dict = merge_dict([inertia_param_dict, friction_param_dict])
    return param_dict


def pseudoinertia_i(pi_dyn_i):
    off_diag = np.array(
        [[0.0, pi_dyn_i[5], pi_dyn_i[7]], [0.0, 0.0, pi_dyn_i[8]], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    I = (
        np.diag(np.array([pi_dyn_i[4], pi_dyn_i[6], pi_dyn_i[9]]))
        + off_diag
        + off_diag.T
    )
    eye3 = np.eye(3)
    Sigma = np.trace(I) / 2.0 * eye3 - I
    J = np.vstack([Sigma, np.reshape(pi_dyn_i[1:4], (1, 3))])
    J = np.hstack(
        [
            J,
            np.reshape(
                np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3], pi_dyn_i[0]]),
                (4, 1),
            ),
        ]
    )
    return J


class DataClass(object):
    def __init__(self, arg_dict):
        for key, value in arg_dict.items():
            if type(value) == str:
                exec(f'self.{key}="{value}"')
            else:
                exec(f"self.{key}={value}")

    def update(self, arg_dict):
        for key, value in arg_dict.items():
            exec(f"self.{key}={value}")


# def logCholeskyVecToParams(theta, pi_prior_i):
#     d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha = theta
#     U = np.exp(alpha) * np.array(
#         [
#             [np.exp(d1), s12, s13, t1],
#             [0, np.exp(d2), s23, t2],
#             [0, 0, np.exp(d3), t3],
#             [0, 0, 0, 1],
#         ]
#     )
#     J0 = inertiaVecToPinertia(pi_prior_i)
#     # J0 = self.J_prior[i]
#     U_J0 = np.dot(U, J0)
#     J = np.dot(U_J0, U.T)
#     pi = pinertiaToVec(J)
#     return pi


def logCholeskyVecToParams(dyn_theta_i):
    d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha = dyn_theta_i
    U = np.exp(alpha) * np.array(
        [
            [np.exp(d1), s12, s13, t1],
            [0, np.exp(d2), s23, t2],
            [0, 0, np.exp(d3), t3],
            [0, 0, 0, 1],
        ]
    )
    J = np.dot(U.T, U)  # lower triangular first
    pi = pinertiaToVec(J)
    return pi


def overparam2pi(dyn_theta):
    nlinks = int(len(dyn_theta) // 10)
    pi = []
    for i in range(nlinks):
        dyn_theta_i = dyn_theta[i * 10 : (i + 1) * 10]
        pi_i = logCholeskyVecToParams(dyn_theta_i)
        pi.extend(pi_i)
    return pi


import scipy


def pi2overparam(dyn_param):
    dyn_theta = []
    njoints = int(len(dyn_param) // 10)
    for i in range(njoints):
        dyn_param_i = dyn_param[i * 10 : (i + 1) * 10]
        Pinertia_i = inertiaVecToPinertia(dyn_param_i)
        U = scipy.linalg.cholesky(Pinertia_i)  # upper cholesky
        # U = np.linalg.cholesky(Pinertia_i) # only lower cholesky
        alpha = np.log(U[3][3])
        U_norm = U / U[3][3]
        d1, d2, d3 = np.log(U_norm[0][0]), np.log(U_norm[1][1]), np.log(U_norm[2][2])
        s12, s13, s23 = U_norm[0][1], U_norm[0][2], U_norm[1][2]
        t1, t2, t3 = U_norm[0][3], U_norm[1][3], U_norm[2][3]
        dyn_theta.extend([d1, d2, d3, s12, s13, s23, t1, t2, t3, alpha])
    return np.array(dyn_theta)


def bregman_regularizer(dyn_param, dyn_param_prior):
    reg = 0.0
    njoints = int(len(dyn_param) // 10)
    for i in range(njoints):
        dyn_param_i = dyn_param[i * 10 : (i + 1) * 10]
        J = inertiaVecToPinertia(dyn_param_i)
        pi_prior_i = dyn_param_prior[i * 10 : (i + 1) * 10]
        J_prior = inertiaVecToPinertia(pi_prior_i)
        _bregman_i = bregman_i(J, J_prior)
        reg += _bregman_i
    return reg


def bregman_i(J, J_prior):
    J1 = J
    J2 = J_prior
    J2inv = np.linalg.inv(J2)
    J2inv_J1 = np.dot(J2inv, J1)
    return (
        -np.log(np.linalg.det(J1))
        + np.log(np.linalg.det(J2))
        + np.trace(J2inv_J1)
        - len(J1)
    )


from system_identification.inertia_model import *


def retrieve_pi_prior_inertia():
    inertia_model = InertiaModel("fr3")
    pi_prior = inertia_model.ref_param
    return pi_prior


def check_CoM_bounded(pi, pi_prior_inertia):
    for i in range(7):
        start, end = i * 10, (i + 1) * 10
        pi_dyn_i = pi[start:end]
        pi_prior_i = pi_prior_inertia[start:end]
        m = pi_dyn_i[0]
        h = pi_dyn_i[1:4]  # here is CoM * mass
        Xs, Qs = inertiaVecToXsQs(pi_prior_i)
        print(h.shape, Xs.shape)
        tmp1 = np.hstack([m, h - m * Xs])
        print(tmp1.shape)
        tmp2 = np.hstack([(h - m * Xs).reshape(3, 1), m * Qs])
        print(tmp2.shape)
        Cpi = np.vstack([tmp1, tmp2])
        print(Cpi.shape)

        w, v = np.linalg.eig(Cpi)
        isSPD = np.all(w > 0)
        if not isSPD:
            return False
    return True
