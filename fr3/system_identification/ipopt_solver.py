import casadi as ca
import numpy as np
import scipy.linalg

from loguru import logger
from system_identification.excitation_generator_new import camera_box_clearance
from system_identification.excitation_optimization import (
    generateAsymFrictionReg,
    generateSymFrictionReg,
)
from system_identification.fourier_utils import flat_params_to_traj


def select_identifiable_columns(regressor, tol=None):
    _, r, pivots = scipy.linalg.qr(regressor, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise ValueError("Regressor has no columns.")

    if tol is None:
        tol = diag.max() * max(regressor.shape) * np.finfo(regressor.dtype).eps

    rank = int(np.sum(diag > tol))
    if rank == 0:
        raise ValueError("Regressor has no identifiable columns.")

    return np.sort(pivots[:rank])


def build_regressor(q, dq, ddq, inertia_model, friction_model=None):
    regressor = inertia_model.regressor(q, dq, ddq)
    if friction_model is None:
        return regressor
    if friction_model == "symmetric":
        return np.hstack((regressor, generateSymFrictionReg(dq)))
    if friction_model == "asymmetric":
        return np.hstack((regressor, generateAsymFrictionReg(dq)))
    raise ValueError(f"Invalid friction model: {friction_model}")


def excitation_eigenvalue_objective(
    regressor,
    eig_eps=1e-9,
    min_eig_weight=1e-6,
):
    min_eig, max_eig = regressor_gramian_extreme_eigs(regressor)
    return (
        np.log(max_eig + eig_eps) - np.log(min_eig + eig_eps) - min_eig_weight * min_eig
    )


def regressor_gramian_extreme_eigs(regressor):
    gramian = regressor.T @ regressor
    gramian = 0.5 * (gramian + gramian.T)
    eigs = np.linalg.eigvalsh(gramian)
    return max(float(eigs[0]), 0.0), max(float(eigs[-1]), 0.0)


def regressor_condition_metrics(regressor, eig_eps=1e-9, min_eig_weight=1e-6):
    min_eig, max_eig = regressor_gramian_extreme_eigs(regressor)
    if min_eig <= 0.0:
        gramian_condition_number = np.inf
        regressor_condition_number = np.inf
    else:
        gramian_condition_number = max_eig / min_eig
        regressor_condition_number = np.sqrt(gramian_condition_number)

    objective = (
        np.log(max_eig + eig_eps) - np.log(min_eig + eig_eps) - min_eig_weight * min_eig
    )
    return {
        "condition_number": regressor_condition_number,
        "gramian_condition_number": gramian_condition_number,
        "min_eig": min_eig,
        "max_eig": max_eig,
        "objective": objective,
    }


def build_params_regressor(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    friction_model=None,
    identifiable_columns=None,
):
    _, q, dq, ddq = flat_params_to_traj(flat_params, fourier_config, robot_config)
    regressor = build_regressor(q, dq, ddq, inertia_model, friction_model)
    if identifiable_columns is not None:
        regressor = regressor[:, identifiable_columns]
    return regressor


def evaluate_params_metrics(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    friction_model=None,
    identifiable_columns=None,
    eig_eps=1e-9,
    min_eig_weight=1e-6,
):
    regressor = build_params_regressor(
        flat_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=friction_model,
        identifiable_columns=identifiable_columns,
    )
    return regressor_condition_metrics(
        regressor,
        eig_eps=eig_eps,
        min_eig_weight=min_eig_weight,
    )


def evaluate_params_objective(
    flat_params,
    fourier_config,
    robot_config,
    inertia_model,
    friction_model=None,
    identifiable_columns=None,
    eig_eps=1e-9,
    min_eig_weight=1e-6,
):
    return evaluate_params_metrics(
        flat_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=friction_model,
        identifiable_columns=identifiable_columns,
        eig_eps=eig_eps,
        min_eig_weight=min_eig_weight,
    )["objective"]


def ipopt_status_is_acceptable(status):
    return status in {
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
        "Maximum_Iterations_Exceeded",
    }


class ExcitationObjectiveCallback(ca.Callback):
    def __init__(
        self,
        name,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=None,
        identifiable_columns=None,
        eig_eps=1e-9,
        min_eig_weight=1e-6,
    ):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._inertia_model = inertia_model
        self._friction_model = friction_model
        self._identifiable_columns = identifiable_columns
        self._eig_eps = eig_eps
        self._min_eig_weight = min_eig_weight
        self._n_params = 2 * fourier_config["order"] * robot_config["njoints"]
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _idx):
        return ca.Sparsity.dense(self._n_params, 1)

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.scalar()

    def eval(self, args):
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        _, q, dq, ddq = flat_params_to_traj(
            flat_params,
            self._fourier_config,
            self._robot_config,
        )
        regressor = build_regressor(
            q, dq, ddq, self._inertia_model, self._friction_model
        )
        if self._identifiable_columns is not None:
            regressor = regressor[:, self._identifiable_columns]
        loss = excitation_eigenvalue_objective(
            regressor,
            eig_eps=self._eig_eps,
            min_eig_weight=self._min_eig_weight,
        )
        return [ca.DM([loss])]


class CameraClearanceCallback(ca.Callback):
    def __init__(self, name, fourier_config, robot_config, stride=5):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._stride = stride
        self._n_params = 2 * fourier_config["order"] * robot_config["njoints"]
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _idx):
        return ca.Sparsity.dense(self._n_params, 1)

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.scalar()

    def eval(self, args):
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self._fourier_config,
            self._robot_config,
        )
        return [ca.DM([camera_box_clearance(q, stride=self._stride)])]


class ConditionNumberIterationCallback(ca.Callback):
    def __init__(
        self,
        name,
        fourier_config,
        robot_config,
        inertia_model,
        n_params,
        n_constraints,
        lbg,
        ubg,
        friction_model=None,
        identifiable_columns=None,
        eig_eps=1e-9,
        min_eig_weight=1e-6,
        early_stop_objective=None,
        early_stop_constraint_tol=1e-6,
        log_metrics=True,
        log_every=1,
        best_condition_initial=np.inf,
        best_candidate_check_every=1,
        best_candidate_callback=None,
    ):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._inertia_model = inertia_model
        self._n_params = n_params
        self._n_constraints = n_constraints
        self._lbg = np.asarray(lbg, dtype=float).reshape(-1)
        self._ubg = np.asarray(ubg, dtype=float).reshape(-1)
        self._friction_model = friction_model
        self._identifiable_columns = identifiable_columns
        self._eig_eps = eig_eps
        self._min_eig_weight = min_eig_weight
        self._early_stop_objective = early_stop_objective
        self._early_stop_constraint_tol = early_stop_constraint_tol
        self._log_metrics = log_metrics
        self._log_every = max(1, int(log_every))
        self._best_candidate_check_every = max(1, int(best_candidate_check_every))
        self._best_candidate_callback = best_candidate_callback
        self._callback_count = 0
        self.stopped_early = False
        self.best_condition_number = float(best_condition_initial)
        self.best_params = None
        self.best_metrics = None
        self.best_callback_count = None
        self.construct(name)

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, idx):
        return ca.nlpsol_out(idx)

    def get_sparsity_in(self, idx):
        name = ca.nlpsol_out(idx)
        if name in ("x", "lam_x"):
            return ca.Sparsity.dense(self._n_params, 1)
        if name in ("g", "lam_g"):
            return ca.Sparsity.dense(self._n_constraints, 1)
        if name == "lam_p":
            return ca.Sparsity.dense(0, 1)
        return ca.Sparsity.scalar()

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.scalar()

    def eval(self, args):
        self._callback_count += 1
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        constraint_values = np.array(args[2], dtype=float).reshape(-1)
        lower_violation = np.maximum(self._lbg - constraint_values, 0.0)
        upper_violation = np.maximum(constraint_values - self._ubg, 0.0)
        max_constraint_violation = float(
            max(np.max(lower_violation), np.max(upper_violation))
        )
        equality_mask = (
            np.isfinite(self._lbg)
            & np.isfinite(self._ubg)
            & np.isclose(self._lbg, self._ubg)
        )
        lower_inequality_mask = np.isfinite(self._lbg) & ~equality_mask
        upper_inequality_mask = np.isfinite(self._ubg) & ~equality_mask
        inequality_margins = []
        if np.any(lower_inequality_mask):
            inequality_margins.append(
                constraint_values[lower_inequality_mask]
                - self._lbg[lower_inequality_mask]
            )
        if np.any(upper_inequality_mask):
            inequality_margins.append(
                self._ubg[upper_inequality_mask]
                - constraint_values[upper_inequality_mask]
            )
        min_inequality_margin = np.inf
        if inequality_margins:
            min_inequality_margin = float(np.min(np.concatenate(inequality_margins)))
        equality_residual = 0.0
        if np.any(equality_mask):
            equality_residual = float(
                np.max(
                    np.abs(constraint_values[equality_mask] - self._lbg[equality_mask])
                )
            )
        constraints_satisfied = (
            max_constraint_violation <= self._early_stop_constraint_tol
        )
        metrics = evaluate_params_metrics(
            flat_params,
            self._fourier_config,
            self._robot_config,
            self._inertia_model,
            friction_model=self._friction_model,
            identifiable_columns=self._identifiable_columns,
            eig_eps=self._eig_eps,
            min_eig_weight=self._min_eig_weight,
        )
        metrics = dict(metrics)
        metrics["min_inequality_margin"] = min_inequality_margin
        metrics["equality_residual"] = equality_residual
        condition_number = float(metrics["condition_number"])
        should_check_best = self._callback_count % self._best_candidate_check_every == 0
        if (
            should_check_best
            and constraints_satisfied
            and np.isfinite(condition_number)
            and condition_number < self.best_condition_number
        ):
            accepted = True
            if self._best_candidate_callback is not None:
                accepted = bool(
                    self._best_candidate_callback(
                        flat_params.copy(),
                        dict(metrics),
                        self._callback_count,
                    )
                )
            if accepted:
                self.best_condition_number = condition_number
                self.best_params = flat_params.copy()
                self.best_metrics = dict(metrics)
                self.best_callback_count = self._callback_count

        if self._log_metrics and self._callback_count % self._log_every == 0:
            logger.info(
                "IPOPT iteration callback "
                f"{self._callback_count}: condition_number={metrics['condition_number']}, "
                f"gramian_condition_number={metrics['gramian_condition_number']}, "
                f"min_eig={metrics['min_eig']}, max_eig={metrics['max_eig']}, "
                f"objective={metrics['objective']}, "
                f"max_constraint_violation={max_constraint_violation}, "
                f"min_inequality_margin={min_inequality_margin}, "
                f"equality_residual={equality_residual}"
            )
        should_stop = (
            self._early_stop_objective is not None
            and metrics["objective"] <= self._early_stop_objective
            and constraints_satisfied
        )
        if should_stop:
            self.stopped_early = True
            logger.info(
                "Early stopping IPOPT: objective "
                f"{metrics['objective']} <= {self._early_stop_objective} and all "
                f"constraints are within tolerance {self._early_stop_constraint_tol}."
            )
        return [ca.DM(1 if should_stop else 0)]


def _joint_param_slices(x, order, njoints):
    start_idx = order * njoints
    for joint_idx in range(njoints):
        a_slice = x[joint_idx * order : (joint_idx + 1) * order]
        b_slice = x[start_idx + joint_idx * order : start_idx + (joint_idx + 1) * order]
        yield a_slice, b_slice


def _scaled_position_limits(robot_config, position_bound_scale):
    upper = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    lower = np.array(robot_config["lower_joint_pos_limits"], dtype=float)

    if len(upper) > 1:
        lower[1] = -1.0
        upper[1] = 1.0

    return lower * position_bound_scale, upper * position_bound_scale


def build_fourier_constraints(
    x,
    fourier_config,
    robot_config,
    camera_clearance_callback=None,
    position_bound_scale=0.8,
    velocity_bound_scale=0.95,
    smooth_eps=1e-12,
):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    omega = 2.0 * np.pi / duration
    harmonic_ids = np.arange(1, order + 1, dtype=float)
    init_pos = np.array(robot_config["init_pos"], dtype=float)
    vel_limit = np.minimum(
        np.array(robot_config["joint_vel_limits"], dtype=float),
        10000.0,
    )
    lower_pos, upper_pos = _scaled_position_limits(robot_config, position_bound_scale)
    offset_limit = np.minimum(upper_pos - init_pos, init_pos - lower_pos)
    if np.any(offset_limit <= 0.0):
        raise ValueError(
            "Initial position is outside the scaled position limits; "
            "reduce position_bound_scale or adjust init_pos."
        )

    g = []
    lbg = []
    ubg = []

    for joint_idx, (a_slice, b_slice) in enumerate(
        _joint_param_slices(x, order, njoints)
    ):
        root = ca.sqrt(a_slice**2 + b_slice**2 + smooth_eps)

        g.append(ca.sum1(a_slice))
        lbg.append(0.0)
        ubg.append(0.0)

        g.append(ca.dot(ca.DM(1.0 / harmonic_ids), b_slice))
        lbg.append(0.0)
        ubg.append(0.0)

        g.append(ca.dot(ca.DM(harmonic_ids), b_slice))
        lbg.append(0.0)
        ubg.append(0.0)

        g.append(ca.sum1(root))
        lbg.append(0.0)
        ubg.append(float(velocity_bound_scale * vel_limit[joint_idx]))

        g.append(ca.dot(ca.DM(1.0 / harmonic_ids), root))
        lbg.append(0.0)
        ubg.append(float(offset_limit[joint_idx] * omega))

    if camera_clearance_callback is not None:
        g.append(camera_clearance_callback(x))
        lbg.append(0.0)
        ubg.append(np.inf)

    return ca.vertcat(*g), np.array(lbg, dtype=float), np.array(ubg, dtype=float)


class IpoptExcitationSolver:
    def __init__(
        self,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=None,
        eig_eps=1e-9,
        min_eig_weight=1e-6,
        camera_collision_stride=5,
        ipopt_max_iter=300,
        ipopt_print_level=5,
        ipopt_hessian_approximation="limited-memory",
        log_condition_every=5,
        early_stop_objective=None,
        early_stop_constraint_tol=1e-6,
        use_identifiable_columns=True,
    ):
        self.fourier_config = fourier_config
        self.robot_config = robot_config
        self.inertia_model = inertia_model
        self.friction_model = friction_model
        self.eig_eps = eig_eps
        self.min_eig_weight = min_eig_weight
        self.camera_collision_stride = camera_collision_stride
        self.ipopt_max_iter = ipopt_max_iter
        self.ipopt_print_level = ipopt_print_level
        self.ipopt_hessian_approximation = ipopt_hessian_approximation
        self.log_condition_every = log_condition_every
        self.early_stop_objective = early_stop_objective
        self.early_stop_constraint_tol = early_stop_constraint_tol
        self.use_identifiable_columns = use_identifiable_columns

    def _initial_identifiable_columns(self, initial_params):
        if not self.use_identifiable_columns:
            return None
        _, q, dq, ddq = flat_params_to_traj(
            initial_params,
            self.fourier_config,
            self.robot_config,
        )
        regressor = build_regressor(q, dq, ddq, self.inertia_model, self.friction_model)
        columns = select_identifiable_columns(regressor)
        logger.info(
            f"Using {len(columns)} of {regressor.shape[1]} identifiable regressor columns."
        )
        return columns

    def solve(self, initial_params):
        initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
        n_params = initial_params.size
        x = ca.MX.sym("fourier_params", n_params)

        identifiable_columns = self._initial_identifiable_columns(initial_params)
        objective_callback = ExcitationObjectiveCallback(
            "excitation_objective",
            self.fourier_config,
            self.robot_config,
            self.inertia_model,
            friction_model=self.friction_model,
            identifiable_columns=identifiable_columns,
            eig_eps=self.eig_eps,
            min_eig_weight=self.min_eig_weight,
        )
        camera_callback = CameraClearanceCallback(
            "camera_clearance",
            self.fourier_config,
            self.robot_config,
            stride=self.camera_collision_stride,
        )

        g, lbg, ubg = build_fourier_constraints(
            x,
            self.fourier_config,
            self.robot_config,
            camera_clearance_callback=camera_callback,
        )
        nlp = {"x": x, "f": objective_callback(x), "g": g}
        solver_options = {
            "ipopt.max_iter": self.ipopt_max_iter,
            "ipopt.print_level": self.ipopt_print_level,
            "ipopt.hessian_approximation": self.ipopt_hessian_approximation,
            "print_time": False,
        }
        iteration_callback = None
        if (
            self.log_condition_every and self.log_condition_every > 0
        ) or self.early_stop_objective is not None:
            iteration_callback = ConditionNumberIterationCallback(
                "condition_number_iteration",
                self.fourier_config,
                self.robot_config,
                self.inertia_model,
                n_params=n_params,
                n_constraints=int(g.shape[0]),
                lbg=lbg,
                ubg=ubg,
                friction_model=self.friction_model,
                identifiable_columns=identifiable_columns,
                eig_eps=self.eig_eps,
                min_eig_weight=self.min_eig_weight,
                early_stop_objective=self.early_stop_objective,
                early_stop_constraint_tol=self.early_stop_constraint_tol,
                log_metrics=bool(
                    self.log_condition_every and self.log_condition_every > 0
                ),
                log_every=max(1, self.log_condition_every),
            )
            solver_options["iteration_callback"] = iteration_callback
            solver_options["iteration_callback_step"] = 1

        solver = ca.nlpsol(
            "excitation_ipopt",
            "ipopt",
            nlp,
            solver_options,
        )
        result = solver(x0=initial_params, lbg=lbg, ubg=ubg)
        x_val = np.array(result["x"], dtype=float).reshape(-1)
        solver_f = float(result["f"])
        final_metrics = evaluate_params_metrics(
            x_val,
            self.fourier_config,
            self.robot_config,
            self.inertia_model,
            friction_model=self.friction_model,
            identifiable_columns=identifiable_columns,
            eig_eps=self.eig_eps,
            min_eig_weight=self.min_eig_weight,
        )
        recomputed_f = final_metrics["objective"]
        best_x = x_val.copy()
        best_metrics = dict(final_metrics)
        best_source = "final"
        if (
            iteration_callback is not None
            and iteration_callback.best_params is not None
            and iteration_callback.best_metrics is not None
            and iteration_callback.best_condition_number
            <= final_metrics["condition_number"]
        ):
            best_x = iteration_callback.best_params.copy()
            best_metrics = dict(iteration_callback.best_metrics)
            best_source = f"iteration_{iteration_callback.best_callback_count}"
        return {
            "x": x_val,
            "f": recomputed_f,
            "solver_f": solver_f,
            "stats": solver.stats(),
            "identifiable_columns": identifiable_columns,
            "best_x": best_x,
            "best_metrics": best_metrics,
            "best_source": best_source,
            "stopped_early": bool(
                iteration_callback is not None and iteration_callback.stopped_early
            ),
        }
