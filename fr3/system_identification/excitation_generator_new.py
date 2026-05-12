from functools import lru_cache

import numpy as np
import pinocchio as pin

from scipy.interpolate import UnivariateSpline as US
from scipy.stats import truncnorm
from system_identification.utils import find_path


@lru_cache(maxsize=None)
def generate_constraints(order: int):
    """
    Trajectories are parameterized as
        q = \sum_k A_k*1/wk sin(wk*t) - B_k*1/wk cos(wk*t)
        qd = \sum_k A_k cos(wk*t) + B_k sin(wk*t)
        qdd = -\sum_k A_k*wk sin(wk*t) + B_k*wk cos(wk*t)

    Initial position, velocity, acceleration constraints: q(0) = qd(0) = qdd(0) = 0
        cA:  sum_k A_1 + ... + A_k + ... + A_L = 0
        cB1: sum_k B_1/1 + ... B_k/k + ... + B_L/L = 0
        cB2: sum_k B_1*1 + ... B_k*k + ... + B_L*L = 0
        where l is the order, so for B, to simplify the
        constraints, we can rewrite the constraints
        [
            [1, 1/2, 1/3, ..., 1/l],
            [1, 2, 3, ..., l]
        ] * B = 0, where B = [B_1, B_2, ..., B_L]
        to the reduced echelon_form so that we can generate
        parameters that satisfy the constraints

    Param:
        order: int, order of the Fourier series
    Return:
        cA: constraints vector, A[0] = -np.dot(cA,A)
        cB: constraints matrix, B[1] = -np.dot(cB[0][2:], B[2:])
                                B[0] = -np.dot(cB[1][1:], B[1:])
    """
    k = np.arange(1, order + 1, dtype=float)
    cA = np.ones(order)
    row0 = 1.0 / k  # [1/1, 1/2, ..., 1/L]
    row1 = k  # [1,   2,   ...,  L]
    diff = row0 - row1
    return cA, np.array([diff / diff[1], row1])


def generate_random_param(order: int, njoints: int):
    """
    Generate constant terms for the Fourier basis, each joint has
    different frequencies

    Param:
        order, njoints
    Return:
        params: an array that includes parameters, shape (2, order, njoints)
            A: parameters for the sin basis, shape (order, njoints)
            B: parameters for the cos basis, shape (order, njoints)
    """
    params = (
        np.random.randn(2, order, njoints)
        * 0.05  # 0.02 works for duration 100, 0.04 works for duration (40, 80], 0.08 works for duration <= 40
        # for single joint, duration 100, 0.08 works
    )  # use tiny variance to tweak the params

    # build constraint coefficients
    cA, cB = generate_constraints(order)
    # get sampled amplitudes for Ak and Bk
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_truncated_param(
    order: int, njoints: int, fourier_config: dict, robot_config: dict
):
    """
    Generate parameters from a truncated gaussian distribution which enforces
    the position and velocities constraints, constraints refer to (14d)-(14e) in
    the paper:
        Parameter Identification of the UA LBR iiwa Robot Including Constraints on
        Physical Feasibility
    """
    # TODO: still have bugs need to be fixed
    duration = fourier_config["duration"]
    oemga_f = 2 * np.pi / duration
    q_max, q_min, qd_max = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    q_max, q_min, qd_max = np.array(q_max), np.array(q_min), np.array(qd_max)
    pos_constant = np.sum([1 / i * np.sqrt(2) for i in range(1, order + 1)])
    pos_upper1 = oemga_f * q_max / pos_constant
    pos_upper2 = -oemga_f * q_min / pos_constant

    vel_constant = order * np.sqrt(2)
    vel_upper = qd_max / vel_constant

    a_max = np.minimum(vel_upper, np.minimum(pos_upper1, pos_upper2))
    a_min = -a_max
    param_flat_shape = 2 * order

    params = []
    for i in range(njoints):
        param = truncnorm.rvs(a_min[i], a_max[i], size=param_flat_shape)
        params.append(param)

    params = np.array(params).T
    params = np.array(params).reshape(2, order, njoints)
    cA, cB = generate_constraints(order)
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_identical_param(order: int, njoints: int):
    """
    Generate constant terms for Fourier basis, each joint has the same
    frequencies
    """
    params = np.random.randn(2, order)
    A_j, B_j = params[0].reshape(-1, 1), params[1].reshape(-1, 1)
    A = np.hstack([A_j for i in range(njoints)])
    B = np.hstack([B_j for i in range(njoints)])

    cA, cB = generate_constraints(order)
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_fourier_traj(
    order, duration, njoints, params, init_pos=None, init_vel=None, fps=100
):
    """Generate trajectories parameterized by Fourier basis"""
    if init_pos is None:
        init_pos = np.zeros(njoints)
    if init_vel is None:
        init_vel = np.zeros(njoints)

    omega_f = 2 * np.pi / duration
    num_samples = int(duration * fps)
    t = np.linspace(0, duration, num_samples + 1)
    A, B = params[0], params[1]  # (order, njoints) each

    omega_k = omega_f * np.arange(1, order + 1)  # (order,)
    inv_ok = 1.0 / omega_k  # (order,)

    # Stack coefficient matrices so q, dq, ddq are all computed in 2 matmuls:
    #   sin_mat @ sin_coeff + cos_mat @ cos_coeff → (T, 3*njoints)
    # Columns [0:J]   → q increment
    # Columns [J:2J]  → dq increment
    # Columns [2J:3J] → ddq increment
    sin_coeff = np.hstack(
        [A * inv_ok[:, None], B, -A * omega_k[:, None]]
    )  # (order, 3J)
    cos_coeff = np.hstack(
        [-B * inv_ok[:, None], A, B * omega_k[:, None]]
    )  # (order, 3J)

    phases = np.outer(t, omega_k)  # (T, order)
    result = np.sin(phases) @ sin_coeff + np.cos(phases) @ cos_coeff  # (T, 3J)

    q = result[:, :njoints] + init_pos
    dq = result[:, njoints : 2 * njoints] + init_vel
    ddq = result[:, 2 * njoints :]
    return t, q, dq, ddq


# def is_traj_valid(q, dq, ddq, robot_config):
#     upperPosLimit, lowerPosLimit, velLimit = (
#         robot_config["upper_joint_pos_limits"],
#         robot_config["lower_joint_pos_limits"],
#         robot_config["joint_vel_limits"],
#     )
#     for q_i, dq_i, ddq_i in zip(q, dq, ddq):
#         for j in range(len(q_i)):
#             if q_i[j] > upperPosLimit[j]:
#                 print(q_i[j], upperPosLimit[j])
#                 print("q upper")
#                 return False
#             elif q_i[j] < lowerPosLimit[j]:
#                 print(q_i[j], lowerPosLimit[j])
#                 print("q lower")
#                 return False
#         if np.any(np.abs(dq_i) - velLimit > 0):
#             print(dq_i, "DQI ==========")
#             print(velLimit, "velLimit ===========")
#             print("dq")
#             return False
#     return True


def is_traj_valid(q, dq, ddq, robot_config):
    upper = np.array(robot_config["upper_joint_pos_limits"])
    lower = np.array(robot_config["lower_joint_pos_limits"])
    vel_limit = np.array(robot_config["joint_vel_limits"])
    return np.all(q <= upper) and np.all(q >= lower) and np.all(np.abs(dq) <= vel_limit)


def is_traj_valid_jointwise(q, dq, ddq, robot_config, joint_id):
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"][joint_id],
        robot_config["lower_joint_pos_limits"][joint_id],
        robot_config["joint_vel_limits"][joint_id],
    )
    for q_i, dq_i, ddq_i in zip(q, dq, ddq):
        for j in range(len(q_i)):
            if q_i[j] > upperPosLimit:
                print(q_i[j], upperPosLimit)
                print("q upper")
                return False
            elif q_i[j] < lowerPosLimit:
                print(q_i[j], lowerPosLimit)
                print("q lower")
                return False
        if np.any(np.abs(dq_i) - velLimit > 0):
            print(dq_i, "DQI ==========")
            print(velLimit, "velLimit ===========")
            print("dq")
            return False
    return True


def is_params_valid(params, fourier_config, robot_config):
    """params: (2, order, njoints) as returned by generate_random_param"""
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    upperPosLimit = np.array(robot_config["upper_joint_pos_limits"])
    velLimit = np.minimum(np.array(robot_config["joint_vel_limits"]), 10000.0)

    A, B = params[0], params[1]  # (order, njoints)
    root_A2B2 = np.sqrt(A**2 + B**2)  # (order, njoints)

    # Velocity bound: sum_k sqrt(Ak_j^2 + Bk_j^2) per joint
    if np.any(root_A2B2.sum(axis=0) > velLimit):
        print("VEL OUT OF BOUND")
        return False

    # Position bound: sum_k sqrt(...)/k per joint
    omega_f = 2 * np.pi / duration
    inv_k = 1.0 / np.arange(1, order + 1, dtype=float)  # (order,)
    if np.any((root_A2B2 * inv_k[:, None]).sum(axis=0) > upperPosLimit * omega_f):
        print("POS OUT OF BOUND")
        return False

    return True


def obtain_fourier_traj(params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    init_pos = robot_config["init_pos"]
    init_vel = robot_config["init_vel"]
    t, q, dq, ddq = generate_fourier_traj(
        order, duration, njoints, params, init_pos, init_vel
    )
    return t, q, dq, ddq


def obtain_valid_traj_param(fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    upper = np.array(robot_config["upper_joint_pos_limits"])
    lower = np.array(robot_config["lower_joint_pos_limits"])

    while True:
        params = generate_random_param(order, njoints)

        # Cheapest check first: analytical param bounds (no trajectory needed)
        if not is_params_valid(params, fourier_config, robot_config):
            continue

        # Generate trajectory only once params pass the analytical check
        t, q, dq, ddq = obtain_fourier_traj(params, fourier_config, robot_config)

        # Position bounds: is_params_valid guarantees velocity and upper position
        # (relative to init_pos); check absolute bounds on the actual trajectory.
        if not (np.all(q <= upper) and np.all(q >= lower)):
            continue

        # Most expensive: FK-based collision check
        no_collision, _ = test_traj(q, dq, ddq)
        if no_collision:
            return t, q, dq, ddq, params


def obtain_valid_traj_param_jointwise(fourier_config, robot_config, joint_id=0):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    duration = fourier_config["duration"]
    init_pos = robot_config["init_pos"]
    init_vel = robot_config["init_vel"]
    valid_traj = False
    njoints = 1
    init_pos_i = [init_pos[joint_id]]
    init_vel_i = [init_vel[joint_id]]
    robot_config["lower_joint_pos_limits"][1] = -0.8  # for not hitting the table
    robot_config["upper_joint_pos_limits"][1] = 0.8
    while not valid_traj:
        params = generate_random_param(order, 1)
        t, q, dq, ddq = generate_fourier_traj(
            order, duration, njoints, params, init_pos_i, init_vel_i
        )
        valid_traj = is_traj_valid_jointwise(q, dq, ddq, robot_config, joint_id)
    return t, q, dq, ddq, params


_pin_model = None
_pin_data = None

_CAMERA_BOX_MARGIN_SCALE = 1.25
_CAMERA_BOX_SPECS_MM = (
    (
        "camera_1_pos_y",
        np.array([165.0, 340.0, 170.0]),
        np.array([170.0, 160.0, 340.0]),
    ),
    (
        "camera_2_pos_y",
        np.array([870.0, 365.0, 180.0]),
        np.array([180.0, 110.0, 360.0]),
    ),
    (
        "camera_1_neg_y",
        np.array([165.0, -340.0, 170.0]),
        np.array([170.0, 160.0, 340.0]),
    ),
    (
        "camera_2_neg_y",
        np.array([870.0, -365.0, 180.0]),
        np.array([180.0, 110.0, 360.0]),
    ),
)
_CAMERA_SAFETY_BOXES_MM = tuple(
    {
        "name": name,
        "center": center,
        "size": size * _CAMERA_BOX_MARGIN_SCALE,
    }
    for name, center, size in _CAMERA_BOX_SPECS_MM
)
_CAMERA_SAFETY_BOXES_M = tuple(
    {
        "name": box["name"],
        "center": box["center"] / 1000.0,
        "half_size": box["size"] / 2000.0,
    }
    for box in _CAMERA_SAFETY_BOXES_MM
)
_CAMERA_SAFETY_BOX_CENTERS_M = np.array(
    [box["center"] for box in _CAMERA_SAFETY_BOXES_M]
)
_CAMERA_SAFETY_BOX_HALF_SIZES_M = np.array(
    [box["half_size"] for box in _CAMERA_SAFETY_BOXES_M]
)
_LINK_COLLISION_SAMPLES = 7


def _get_pin_model_data():
    global _pin_model, _pin_data
    if _pin_model is None:
        urdf_file = find_path("fr3.urdf", "../robot_description")
        _pin_model = pin.buildModelFromUrdf(urdf_file)
        _pin_data = _pin_model.createData()
    return _pin_model, _pin_data


def _point_inside_box(point, box):
    return np.all(np.abs(point[:2] - box["center"][:2]) <= box["half_size"][:2])


def _sample_robot_body_points(model, data):
    joint_positions = [
        data.oMi[joint_id].translation.copy() for joint_id in range(1, model.njoints)
    ]
    if not joint_positions:
        return []

    points = [joint_positions[0]]
    for start, end in zip(joint_positions[:-1], joint_positions[1:]):
        for alpha in np.linspace(0.0, 1.0, _LINK_COLLISION_SAMPLES, endpoint=True)[1:]:
            points.append((1.0 - alpha) * start + alpha * end)
    return points


def camera_box_clearance(q, stride=1):
    """Return minimum signed XY clearance from sampled robot body points to camera boxes.

    Positive means the point is outside the XY rectangle, zero is on a box edge,
    and negative means the point projection is inside the XY rectangle.
    """
    model, data = _get_pin_model_data()
    stride = max(1, int(stride))
    min_clearance = np.inf

    for q_i in q[::stride]:
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)
        body_points = np.array(_sample_robot_body_points(model, data))
        if body_points.size == 0:
            continue

        delta = (
            np.abs(body_points[:, None, :2] - _CAMERA_SAFETY_BOX_CENTERS_M[None, :, :2])
            - _CAMERA_SAFETY_BOX_HALF_SIZES_M[None, :, :2]
        )
        outside_delta = np.maximum(delta, 0.0)
        signed_distance = np.linalg.norm(outside_delta, axis=2) + np.minimum(
            np.max(delta, axis=2),
            0.0,
        )
        min_clearance = min(min_clearance, float(np.min(signed_distance)))

    return min_clearance


def test_traj(q, dq, _ddq):
    model, data = _get_pin_model_data()

    # Vectorized limit checks before the FK loop
    if np.any(np.abs(dq) > model.velocityLimit):
        return False, None
    if np.any(np.abs(q) > model.upperPositionLimit):
        return False, None

    ee_pos = []
    for q_i in q:
        pin.forwardKinematics(model, data, q_i)
        ee_pos_i = data.oMi[-1].translation.copy()

        if ee_pos_i[0] < -0.5 or abs(ee_pos_i[1]) > 0.8 or ee_pos_i[2] < 0.2:
            return False, None

        ee_pos.append(ee_pos_i)

    return True, ee_pos


def test_traj_collision(q, dq, _ddq):
    model, data = _get_pin_model_data()

    if np.any(np.abs(dq) > model.velocityLimit):
        return False, None
    if np.any(np.abs(q) > model.upperPositionLimit):
        return False, None

    ee_pos = []
    for q_i in q:
        pin.forwardKinematics(model, data, q_i)
        pin.updateFramePlacements(model, data)
        ee_pos_i = data.oMi[-1].translation.copy()

        if ee_pos_i[0] < 0.0 or abs(ee_pos_i[1]) > 0.8 or ee_pos_i[2] < 0.2:
            return False, None

        for body_point in _sample_robot_body_points(model, data):
            if any(
                _point_inside_box(body_point, box) for box in _CAMERA_SAFETY_BOXES_M
            ):
                return False, None

        ee_pos.append(ee_pos_i)

    return True, ee_pos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Excitation generator")
    parser.add_argument("--robot", type=str, default="iiwas")
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=80)
    args = parser.parse_args()

    # setup gym env
    from system_identification.my_utils import retrieve_robot_config

    robot_config = retrieve_robot_config(args.robot)
    njoints = robot_config["njoints"]

    # ===================================================================================
    # Generate exciting trajectories
    # ===================================================================================
    # parameterize trajectories as fourier series and initialize the parameters
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}

    # t, qs, qds, qdds, init_params = obtain_valid_traj_param(
    #     fourier_config, robot_config
    # )
    # print(t.shape, qs.shape, qds.shape, qdds.shape, init_params.shape)

    joint_id = 0
    t, qs, qds, qdds, init_params = obtain_valid_traj_param_jointwise(
        fourier_config, robot_config, joint_id=joint_id
    )

    import matplotlib.pyplot as plt

    from system_identification.my_utils import vis_compare_seqs

    vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
    print(t.shape, qs.shape, qds.shape, qdds.shape, init_params.shape)
    init_dict = {"t": t, "q": qs, "dq": qds, "ddq": qdds, "init_params": init_params}
    if joint_id != -1:
        np.save(
            f"../experiments/traj_data/init_params_{args.fourier_duration}_{joint_id}.npy",
            init_dict,
            allow_pickle=True,
        )
    else:
        np.save(
            f"../experiments/traj_data/init_params_{args.fourier_duration}.npy",
            init_dict,
            allow_pickle=True,
        )
    plt.show()
