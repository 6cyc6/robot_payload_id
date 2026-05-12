from datetime import datetime

from loguru import logger
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from system_identification.excitation_generator import *
from system_identification.inertia_model import InertiaModel
from system_identification.utils import *


def params2cond(params, fourier_config, robot_config, sysID):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to constuct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    _, cond_num, s = QR_dim_reduction(regressor)
    logger.info(f"current singular value: {s}, condition number: {cond_num}")
    return cond_num


def generateSymFrictionReg(dq, vbrk=0.001):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vst = vbrk * np.sqrt(2)
    vcoul = vbrk * 2
    feature_Fbrk_mins_Fc = np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst
    feature_Fc = np.tanh(dq / vcoul)
    feature_viscous = dq
    features = [feature_Fbrk_mins_Fc, feature_Fc, feature_viscous]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg


def generateAsymFrictionReg(dq, vbrk=0.001):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vst = vbrk * np.sqrt(2)
    vcoul = vbrk * 2
    feature_Fbrk_mins_Fc_pos = (
        np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst * (dq > 0)
    )
    feature_Fc_pos = np.tanh(dq / vcoul) * (dq > 0)
    feature_viscous_pos = dq * (dq > 0)
    feature_Fbrk_mins_Fc_neg = (
        np.sqrt(2 * np.e) * np.exp(-((dq / vst) ** 2)) * dq / vst * (dq < 0)
    )
    feature_Fc_neg = np.tanh(dq / vcoul) * (dq < 0)
    feature_viscous_neg = dq * (dq < 0)
    features = [
        feature_Fbrk_mins_Fc_pos,
        feature_Fc_pos,
        feature_viscous_pos,
        feature_Fbrk_mins_Fc_neg,
        feature_Fc_neg,
        feature_viscous_neg,
    ]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg


def params2condFriction(params, fourier_config, robot_config, sysID, friction_model):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to constuct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    if friction_model == "symmetric":
        regressorFriction = generateSymFrictionReg(qds)
    elif friction_model == "asymmetric":
        regressorFriction = generateAsymFrictionReg(qds)
    else:
        raise ValueError("Invalid friction model")
    regressor = np.hstack([regressor, regressorFriction])
    _, cond_num, s = QR_dim_reduction(regressor)
    logger.info(f"current singular value: {s}, condition number: {cond_num}")
    return cond_num


def params2coverage(params, fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to construct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    qs_min = np.min(qs, axis=0)
    qs_max = np.max(qs, axis=0)
    loss = -np.sum((qs_max - qs_min) ** 2)
    return loss


def constraints(flatten_params, fourier_config, robot_config, optimizer):
    """
    Return constraints for parameter optimization, including
        1) Linear constraints: Initial position, velocity, accleration constraints: q(0)=init_pos, dq(0)=ddq(0)=0
        2) Nonlinear constraints: Extreme position, velocity constraints
        3) Parameter constraints

        The linear equality constraints can express as M.dot(flatten params), while M are the constant of the linear
        equality constraints, organized as a matrix, for a 2-dof robot with a fourier series trajectory of order 5,
        the position constraints matrx is
            [
        M=      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 1/1, 1/2., 1/3., 1/4., 1/5., 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 0, 0, 0, 0, 0, 1/1, 1/2., 1/3., 1/4., 1/5.],
            ]
        M.dot(params) = [
                            0,
                            0,
                        ]

    Refer to paper: Parameter Identification of the KUKA LBR iiwa Robot Including
        Constraints on Physical Feasibility
    """
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    init_pos = robot_config["init_pos"]
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    upperPosLimit, lowerPosLimit, velLimit = (
        np.array(upperPosLimit),
        np.array(lowerPosLimit),
        np.array(velLimit),
    )

    # set to 95 of the limit to avoid violation
    bound = 0.8
    lowerPosLimit[1] = -1.0
    upperPosLimit[1] = 1.0
    upperPosLimit = upperPosLimit * bound
    lowerPosLimit = lowerPosLimit * bound
    logger.info(f"upperPosLimit: {upperPosLimit}")
    logger.info(f"lowerPosLimit: {lowerPosLimit}")
    logger.info(f"velLimit: {velLimit}")
    omega_f = 2 * np.pi / duration

    # linear constraints for initial positions, velocities and accelerations
    init_vel_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    for i in range(njoints):
        init_vel_constraint_mat[i, i * order : (i + 1) * order] = np.ones(order)
    vel_lb = vel_ub = np.zeros(njoints)
    keep_feasible = np.array([True for i in range(njoints)])
    init_vel_constraint = LinearConstraint(
        init_vel_constraint_mat, vel_lb, vel_ub, keep_feasible=keep_feasible
    )

    start_idx = flatten_params.shape[0] // 2
    init_pos_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    const = np.array([1 / i for i in range(1, order + 1)])
    for i in range(njoints):
        init_pos_constraint_mat[
            i, start_idx + i * order : start_idx + (i + 1) * order
        ] = const
    pos_lb = pos_ub = np.array(init_pos)
    init_pos_constraint = LinearConstraint(
        init_pos_constraint_mat, pos_lb, pos_ub, keep_feasible=keep_feasible
    )

    init_acc_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    const = np.array([i for i in range(1, order + 1)])
    for i in range(njoints):
        init_acc_constraint_mat[
            i, start_idx + i * order : start_idx + (i + 1) * order
        ] = const
    acc_lb = acc_ub = np.zeros(njoints)
    init_acc_constraint = LinearConstraint(
        init_acc_constraint_mat, acc_lb, acc_ub, keep_feasible=keep_feasible
    )

    # nonlinear constraints for positions and velocities
    extreme_constraint_mat = np.zeros((njoints, order * njoints))
    for i in range(njoints):
        extreme_constraint_mat[i, i * order : (i + 1) * order] = np.ones(order)

    def extreme_vel_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        return np.dot(extreme_constraint_mat, root_A2B2)

    extreme_vel_lb = -velLimit * 0.95
    extreme_vel_ub = (
        velLimit * 0.95
    )  # here use a tighter bound for velocity to optimize
    # such that we can guarantee that after optimization
    # all trajectories are within the bound
    for i in range(len(extreme_vel_ub)):
        if extreme_vel_ub[i] > 10000:
            extreme_vel_ub[i] = 10000
    extreme_vel_constraint = NonlinearConstraint(
        extreme_vel_func, extreme_vel_lb, extreme_vel_ub
    )

    def extreme_pos_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        divider = np.array([1 / i for i in range(1, order + 1)] * njoints)
        root_A2B2_divided = root_A2B2 * divider
        return np.dot(extreme_constraint_mat, root_A2B2_divided)

    extreme_pos_lb = lowerPosLimit * omega_f
    extreme_pos_ub = upperPosLimit * omega_f
    extreme_pos_constraint = NonlinearConstraint(
        extreme_pos_func, extreme_pos_lb, extreme_pos_ub
    )

    # linear constraints for parameters
    const = np.array([i / order * omega_f for i in range(1, order + 1)] * njoints)
    const = np.hstack([const, const])

    q_max_extend = np.repeat(upperPosLimit, order)
    q_max_extend = np.hstack([q_max_extend, q_max_extend])

    dq_max_extend = np.repeat(velLimit, order)
    dq_max_extend = np.hstack([dq_max_extend, dq_max_extend])
    dq_min_extend = -dq_max_extend

    param_ub = np.minimum(const * q_max_extend, dq_max_extend)
    q_min_extend = np.repeat(lowerPosLimit, order)
    q_min_extend = np.hstack([q_min_extend, q_min_extend])
    param_lb = np.maximum(const * q_min_extend, dq_min_extend)

    keep_feasible = [True for i in range(flatten_params.shape[0])]
    A = np.identity(flatten_params.shape[0])
    extreme_param_constraint = LinearConstraint(
        A, param_lb, param_ub, keep_feasible=keep_feasible
    )

    # equality constraints for initial positions, velocities and accelerations
    init_vel_constraint_func = lambda x: np.dot(init_vel_constraint_mat, x)
    init_pos_constraint_func = lambda x: np.dot(init_pos_constraint_mat, x)
    init_acc_constraint_func = lambda x: np.dot(init_acc_constraint_mat, x)

    # inequality constraints for upper and lower bounds of positions and velocities
    upper_pos_constraint_func = lambda x: -(extreme_pos_func(x) - extreme_pos_ub)
    upper_vel_constraint_func = lambda x: -(extreme_vel_func(x) - extreme_vel_ub)
    lower_pos_constraint_func = lambda x: -extreme_pos_func(x) - extreme_pos_lb

    # inequality constraints for parameters from the paper, but it seems these constraints
    # are useless so we ommited them, according to the paper, I can't find the parameters
    # that satisfy with the joint limit constraints but not satisfy with the parameter
    # constraints
    # the parameter constraints are tighter bound compare with joint limit constraints
    # and it seems they are redundant compare with constraints in (14d) and (14e) in the paper

    extreme_param_upper_constraint_func = lambda x: -np.dot(A, x) + param_ub
    extreme_param_lower_constraint_func = lambda x: np.dot(A, x) - param_lb

    if optimizer == "SLSQP":
        # SLSQP seems to have problems with the constraints, it's not able to find a feasible solution
        cons = (
            {"type": "eq", "fun": init_vel_constraint_func},
            {"type": "eq", "fun": init_pos_constraint_func},
            {"type": "eq", "fun": init_acc_constraint_func},
            # {"type": "ineq", "fun": upper_pos_constraint_func},
            # {"type": "ineq", "fun": upper_vel_constraint_func},
            # {"type": "ineq", "fun": lower_pos_constraint_func},
            {"type": "ineq", "fun": extreme_param_upper_constraint_func},
            {"type": "ineq", "fun": extreme_param_lower_constraint_func},
        )
    elif optimizer == "trust-constr":
        cons = (
            init_pos_constraint,
            init_vel_constraint,
            init_acc_constraint,
            extreme_pos_constraint,
            extreme_vel_constraint,
            # extreme_param_constraint, # would make the optimization infeasible, it's a tighter bound compare with the joint limit constraints
        )
    return cons
