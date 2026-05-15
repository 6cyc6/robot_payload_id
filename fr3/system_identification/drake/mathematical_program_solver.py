import numpy as np

from loguru import logger
from pydrake.all import IpoptSolver, MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd, ExtractGradient, ExtractValue
from system_identification.drake.camera_collision import DrakeCameraCollisionChecker
from system_identification.fourier_utils import flat_params_to_traj
from system_identification.ipopt_solver import (
    build_regressor,
    evaluate_params_metrics,
    select_identifiable_columns,
)


def fourier_constraint_bounds(
    fourier_config,
    robot_config,
    *,
    velocity_limit_scale=0.95,
    position_limit_scale=0.95,
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
    upper_pos = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    lower_pos = np.array(robot_config["lower_joint_pos_limits"], dtype=float)
    if len(upper_pos) > 1:
        lower_pos[1] = -1.0
        upper_pos[1] = 1.0
    velocity_limit_scale = float(velocity_limit_scale)
    position_limit_scale = float(position_limit_scale)
    lower_pos *= position_limit_scale
    upper_pos *= position_limit_scale
    offset_limit = np.minimum(upper_pos - init_pos, init_pos - lower_pos)
    if np.any(offset_limit <= 0.0):
        raise ValueError(
            "Initial position is outside the scaled position limits; "
            "reduce the position bound scale or adjust init_pos."
        )

    lbg = []
    ubg = []
    for joint_idx in range(njoints):
        lbg.extend((0.0, 0.0, 0.0, 0.0, 0.0))
        ubg.extend(
            (
                0.0,
                0.0,
                0.0,
                float(velocity_limit_scale * vel_limit[joint_idx]),
                float(offset_limit[joint_idx] * omega),
            )
        )
    return np.asarray(lbg, dtype=float), np.asarray(ubg, dtype=float)


def fourier_constraint_values(
    flat_params, fourier_config, robot_config, smooth_eps=1e-12
):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    harmonic_ids = np.arange(1, order + 1, dtype=float)
    flat_params = np.asarray(flat_params, dtype=float).reshape(-1)
    a_params = flat_params[: order * njoints].reshape(njoints, order).T
    b_params = flat_params[order * njoints :].reshape(njoints, order).T
    root = np.sqrt(a_params**2 + b_params**2 + smooth_eps)

    values = []
    for joint_idx in range(njoints):
        a = a_params[:, joint_idx]
        b = b_params[:, joint_idx]
        r = root[:, joint_idx]
        values.extend(
            (
                np.sum(a),
                np.dot(1.0 / harmonic_ids, b),
                np.dot(harmonic_ids, b),
                np.sum(r),
                np.dot(1.0 / harmonic_ids, r),
            )
        )
    return np.asarray(values, dtype=float)


def _has_autodiff(values):
    values = np.asarray(values, dtype=object).reshape(-1)
    return values.size > 0 and isinstance(values[0], AutoDiffXd)


def _double_vector(values):
    if _has_autodiff(values):
        return np.asarray(ExtractValue(values), dtype=float).reshape(-1)
    return np.asarray(values, dtype=float).reshape(-1)


def _input_derivatives(values):
    return np.asarray(ExtractGradient(values), dtype=float).reshape(
        len(_double_vector(values)),
        -1,
    )


def _finite_difference_jacobian(func, x, step=1e-6):
    x = np.asarray(x, dtype=float).reshape(-1)
    f0 = np.asarray(func(x), dtype=float).reshape(-1)
    jacobian = np.empty((f0.size, x.size), dtype=float)
    for idx in range(x.size):
        h = step * max(1.0, abs(x[idx]))
        x_step = x.copy()
        x_step[idx] += h
        jacobian[:, idx] = (np.asarray(func(x_step), dtype=float).reshape(-1) - f0) / h
    return f0, jacobian


def _autodiff_scalar(func, values, step=1e-6):
    if not _has_autodiff(values):
        return float(func(_double_vector(values)))

    x = _double_vector(values)
    f0, jacobian = _finite_difference_jacobian(
        lambda z: np.array([func(z)], dtype=float),
        x,
        step=step,
    )
    derivatives = jacobian[0] @ _input_derivatives(values)
    return AutoDiffXd(float(f0[0]), derivatives)


def _autodiff_vector(func, values, step=1e-6):
    if not _has_autodiff(values):
        return np.asarray(func(_double_vector(values)), dtype=float).reshape(-1)

    x = _double_vector(values)
    f0, jacobian = _finite_difference_jacobian(func, x, step=step)
    input_derivatives = _input_derivatives(values)
    return np.asarray(
        [
            AutoDiffXd(float(value), jacobian[row_idx] @ input_derivatives)
            for row_idx, value in enumerate(f0)
        ],
        dtype=object,
    )


class DrakeMathematicalProgramExcitationSolver:
    def __init__(
        self,
        fourier_config,
        robot_config,
        inertia_model,
        robot_name="fr3",
        friction_model=None,
        eig_eps=1e-9,
        min_eig_weight=1e-6,
        camera_collision_stride=3,
        ipopt_max_iter=300,
        ipopt_print_level=5,
        ipopt_hessian_approximation="limited-memory",
        log_condition_every=5,
        early_stop_constraint_tol=1e-6,
        use_identifiable_columns=True,
        drake_min_distance=0.02,
        drake_robot_sphere_radius=0.05,
        drake_robot_link_samples=5,
        drake_camera_chamfer_radius=0.02,
        drake_xy_prism_height=None,
        link_y_lower=-0.35,
        link_y_upper=0.35,
        use_link_y_bounds=True,
        best_condition_initial=1.0e7,
        best_candidate_check_every=3,
        best_candidate_callback=None,
        best_candidate_fourier_velocity_margin_tolerance=0.0,
        best_candidate_fourier_position_margin_tolerance=0.0,
        best_candidate_fourier_velocity_margin_min=None,
        best_candidate_fourier_position_margin_min=None,
        fourier_velocity_limit_scale=0.95,
        fourier_position_limit_scale=0.95,
        finite_difference_step=1e-6,
        collision_checker=None,
    ):
        self.fourier_config = fourier_config
        self.robot_config = robot_config
        self.inertia_model = inertia_model
        self.robot_name = robot_name
        self.friction_model = friction_model
        self.eig_eps = eig_eps
        self.min_eig_weight = min_eig_weight
        self.camera_collision_stride = max(1, int(camera_collision_stride))
        self.ipopt_max_iter = ipopt_max_iter
        self.ipopt_print_level = ipopt_print_level
        self.ipopt_hessian_approximation = ipopt_hessian_approximation
        self.log_condition_every = log_condition_every
        self.early_stop_constraint_tol = early_stop_constraint_tol
        self.use_identifiable_columns = use_identifiable_columns
        self.link_y_lower = link_y_lower
        self.link_y_upper = link_y_upper
        self.use_link_y_bounds = use_link_y_bounds
        self.best_condition_initial = best_condition_initial
        self.best_candidate_check_every = max(1, int(best_candidate_check_every))
        self.best_candidate_callback = best_candidate_callback
        if best_candidate_fourier_velocity_margin_min is None:
            best_candidate_fourier_velocity_margin_min = -max(
                0.0,
                float(best_candidate_fourier_velocity_margin_tolerance),
            )
        if best_candidate_fourier_position_margin_min is None:
            best_candidate_fourier_position_margin_min = -max(
                0.0,
                float(best_candidate_fourier_position_margin_tolerance),
            )
        self.best_candidate_fourier_velocity_margin_min = float(
            best_candidate_fourier_velocity_margin_min
        )
        self.best_candidate_fourier_position_margin_min = float(
            best_candidate_fourier_position_margin_min
        )
        self.fourier_velocity_limit_scale = float(fourier_velocity_limit_scale)
        self.fourier_position_limit_scale = float(fourier_position_limit_scale)
        self.finite_difference_step = finite_difference_step
        self.collision_checker = collision_checker or DrakeCameraCollisionChecker(
            robot_name=robot_name,
            min_distance=drake_min_distance,
            robot_sphere_radius=drake_robot_sphere_radius,
            robot_link_samples=drake_robot_link_samples,
            camera_chamfer_radius=drake_camera_chamfer_radius,
            xy_prism_height=drake_xy_prism_height,
        )

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

    def _objective(self, identifiable_columns, flat_params):
        flat_params = np.asarray(flat_params, dtype=float).reshape(-1)
        return float(
            evaluate_params_metrics(
                flat_params,
                self.fourier_config,
                self.robot_config,
                self.inertia_model,
                friction_model=self.friction_model,
                identifiable_columns=identifiable_columns,
                eig_eps=self.eig_eps,
                min_eig_weight=self.min_eig_weight,
            )["objective"]
        )

    def _path_constraint_values(self, flat_params):
        flat_params = np.asarray(flat_params, dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self.fourier_config,
            self.robot_config,
        )
        q_sampled = q[:: self.camera_collision_stride]
        values = [self.collision_checker.minimum_distance_constraint_values(q_sampled)]
        if self.use_link_y_bounds:
            values.append(
                self.collision_checker.robot_link_y_margins(
                    q_sampled,
                    lower=self.link_y_lower,
                    upper=self.link_y_upper,
                ).reshape(-1)
            )
        return np.concatenate(values)

    def _objective_callback(self, identifiable_columns, flat_params):
        return _autodiff_scalar(
            lambda z: self._objective(identifiable_columns, z),
            flat_params,
            step=self.finite_difference_step,
        )

    def _fourier_constraint_callback(self, flat_params):
        return _autodiff_vector(
            lambda z: fourier_constraint_values(
                z, self.fourier_config, self.robot_config
            ),
            flat_params,
            step=self.finite_difference_step,
        )

    def _path_constraint_callback(self, flat_params):
        return _autodiff_vector(
            self._path_constraint_values,
            flat_params,
            step=self.finite_difference_step,
        )

    def _path_constraint_bounds(self):
        n_samples = len(
            range(
                0,
                int(self.fourier_config["duration"] * 100) + 1,
                self.camera_collision_stride,
            )
        )
        n_outputs = n_samples * (3 if self.use_link_y_bounds else 1)
        lower = np.zeros(n_outputs, dtype=float)
        upper = np.full(n_outputs, np.inf, dtype=float)
        upper[:n_samples] = 1.0
        return lower, upper

    @staticmethod
    def _max_constraint_violation(values, lower, upper):
        lower_violation = np.maximum(lower - values, 0.0)
        upper_violation = np.maximum(values - upper, 0.0)
        return float(max(np.max(lower_violation), np.max(upper_violation)))

    @staticmethod
    def _min_or_inf(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            return np.inf
        return float(np.min(values))

    def _constraint_margin_report(
        self,
        fourier_values,
        fourier_lower,
        fourier_upper,
        path_values,
        path_lower,
        path_upper,
    ):
        njoints = int(self.robot_config["njoints"])
        fourier_values = np.asarray(fourier_values, dtype=float).reshape(njoints, 5)
        fourier_lower = np.asarray(fourier_lower, dtype=float).reshape(njoints, 5)
        fourier_upper = np.asarray(fourier_upper, dtype=float).reshape(njoints, 5)

        fourier_equality_residual = float(np.max(np.abs(fourier_values[:, :3])))
        fourier_velocity_margin = self._min_or_inf(
            np.minimum(
                fourier_values[:, 3] - fourier_lower[:, 3],
                fourier_upper[:, 3] - fourier_values[:, 3],
            )
        )
        fourier_position_margin = self._min_or_inf(
            np.minimum(
                fourier_values[:, 4] - fourier_lower[:, 4],
                fourier_upper[:, 4] - fourier_values[:, 4],
            )
        )

        path_values = np.asarray(path_values, dtype=float).reshape(-1)
        path_lower = np.asarray(path_lower, dtype=float).reshape(-1)
        path_upper = np.asarray(path_upper, dtype=float).reshape(-1)
        n_collision = int(np.count_nonzero(np.isfinite(path_upper)))
        collision_values = path_values[:n_collision]
        collision_lower = path_lower[:n_collision]
        collision_upper = path_upper[:n_collision]
        drake_collision_margin = self._min_or_inf(
            np.minimum(
                collision_values - collision_lower,
                collision_upper - collision_values,
            )
        )
        link_y_margin = self._min_or_inf(
            path_values[n_collision:] - path_lower[n_collision:]
        )

        return {
            "min_inequality_margin": min(
                fourier_velocity_margin,
                fourier_position_margin,
                drake_collision_margin,
                link_y_margin,
            ),
            "fourier_equality_residual": fourier_equality_residual,
            "fourier_velocity_margin": fourier_velocity_margin,
            "fourier_position_margin": fourier_position_margin,
            "drake_collision_margin": drake_collision_margin,
            "link_y_margin": link_y_margin,
        }

    def solve(self, initial_params):
        initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
        n_params = initial_params.size
        identifiable_columns = self._initial_identifiable_columns(initial_params)

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(n_params, "fourier_params")
        prog.AddCost(
            lambda z: self._objective_callback(identifiable_columns, z),
            vars=x,
        )

        fourier_lbg, fourier_ubg = fourier_constraint_bounds(
            self.fourier_config,
            self.robot_config,
            velocity_limit_scale=self.fourier_velocity_limit_scale,
            position_limit_scale=self.fourier_position_limit_scale,
        )
        prog.AddConstraint(
            self._fourier_constraint_callback,
            fourier_lbg,
            fourier_ubg,
            vars=x,
        )

        path_lbg, path_ubg = self._path_constraint_bounds()
        prog.AddConstraint(self._path_constraint_callback, path_lbg, path_ubg, vars=x)
        prog.SetInitialGuess(x, initial_params)

        callback_state = {
            "count": 0,
            "best_condition_number": float(self.best_condition_initial),
            "best_params": None,
            "best_metrics": None,
            "best_callback_count": None,
        }

        def iteration_callback(z):
            callback_state["count"] += 1
            count = callback_state["count"]
            z = np.asarray(z, dtype=float).reshape(-1)
            metrics = evaluate_params_metrics(
                z,
                self.fourier_config,
                self.robot_config,
                self.inertia_model,
                friction_model=self.friction_model,
                identifiable_columns=identifiable_columns,
                eig_eps=self.eig_eps,
                min_eig_weight=self.min_eig_weight,
            )
            fourier_values = fourier_constraint_values(
                z,
                self.fourier_config,
                self.robot_config,
            )
            path_values = self._path_constraint_values(z)
            fourier_violation = self._max_constraint_violation(
                fourier_values,
                fourier_lbg,
                fourier_ubg,
            )
            path_violation = self._max_constraint_violation(
                path_values,
                path_lbg,
                path_ubg,
            )
            max_violation = max(fourier_violation, path_violation)
            margin_report = self._constraint_margin_report(
                fourier_values,
                fourier_lbg,
                fourier_ubg,
                path_values,
                path_lbg,
                path_ubg,
            )
            metrics = dict(metrics)
            metrics.update(margin_report)
            should_log_condition = bool(
                self.log_condition_every and count % self.log_condition_every == 0
            )
            should_check_best = count % self.best_candidate_check_every == 0

            condition_number = float(metrics["condition_number"])
            best_candidate_callback_ran = False
            if should_check_best:
                best_check_failure_reasons = []
                constraint_tol = max(0.0, float(self.early_stop_constraint_tol))
                if margin_report["fourier_equality_residual"] > constraint_tol:
                    best_check_failure_reasons.append(
                        "fourier_equality_residual="
                        f"{margin_report['fourier_equality_residual']} > "
                        f"{constraint_tol}"
                    )
                if (
                    margin_report["fourier_velocity_margin"]
                    < self.best_candidate_fourier_velocity_margin_min
                ):
                    best_check_failure_reasons.append(
                        "fourier_velocity_margin="
                        f"{margin_report['fourier_velocity_margin']} < "
                        f"{self.best_candidate_fourier_velocity_margin_min}"
                    )
                if (
                    margin_report["fourier_position_margin"]
                    < self.best_candidate_fourier_position_margin_min
                ):
                    best_check_failure_reasons.append(
                        "fourier_position_margin="
                        f"{margin_report['fourier_position_margin']} < "
                        f"{self.best_candidate_fourier_position_margin_min}"
                    )
                if margin_report["drake_collision_margin"] < -constraint_tol:
                    best_check_failure_reasons.append(
                        "drake_collision_margin="
                        f"{margin_report['drake_collision_margin']} < "
                        f"{-constraint_tol}"
                    )
                if margin_report["link_y_margin"] < -constraint_tol:
                    best_check_failure_reasons.append(
                        "link_y_margin="
                        f"{margin_report['link_y_margin']} < "
                        f"{-constraint_tol}"
                    )
                if not np.isfinite(condition_number):
                    best_check_failure_reasons.append(
                        f"condition_number_not_finite={condition_number}"
                    )
                elif condition_number >= callback_state["best_condition_number"]:
                    best_check_failure_reasons.append(
                        "condition_number_not_better="
                        f"{condition_number} >= "
                        f"{callback_state['best_condition_number']}"
                    )

                if not best_check_failure_reasons:
                    accepted = True
                    if self.best_candidate_callback is not None:
                        best_candidate_callback_ran = True
                        accepted = bool(
                            self.best_candidate_callback(z.copy(), dict(metrics), count)
                        )
                    if accepted:
                        callback_state["best_condition_number"] = condition_number
                        callback_state["best_params"] = z.copy()
                        callback_state["best_metrics"] = dict(metrics)
                        callback_state["best_callback_count"] = count
                else:
                    logger.info(
                        "Best-condition check FAIL at Drake IPOPT callback "
                        f"{count}: "
                        f"{'; '.join(best_check_failure_reasons)}; "
                        f"condition_number={condition_number}; "
                        f"current_best_condition_number="
                        f"{callback_state['best_condition_number']}; "
                        f"max_constraint_violation={max_violation}; "
                        f"min_inequality_margin="
                        f"{margin_report['min_inequality_margin']}"
                    )

            if should_log_condition and not best_candidate_callback_ran:
                logger.info(
                    "Drake IPOPT callback "
                    f"{count}: condition_number={metrics['condition_number']}, "
                    f"gramian_condition_number={metrics['gramian_condition_number']}, "
                    f"min_eig={metrics['min_eig']}, max_eig={metrics['max_eig']}, "
                    f"objective={metrics['objective']}, "
                    f"max_constraint_violation={max_violation}, "
                    f"min_inequality_margin={margin_report['min_inequality_margin']}, "
                    f"fourier_equality_residual="
                    f"{margin_report['fourier_equality_residual']}, "
                    f"fourier_velocity_margin="
                    f"{margin_report['fourier_velocity_margin']}, "
                    f"fourier_position_margin="
                    f"{margin_report['fourier_position_margin']}, "
                    f"drake_collision_margin="
                    f"{margin_report['drake_collision_margin']}, "
                    f"link_y_margin={margin_report['link_y_margin']}"
                )

        if (
            self.log_condition_every
            or self.best_candidate_callback is not None
            or self.best_candidate_check_every
        ):
            prog.AddVisualizationCallback(iteration_callback, x)

        solver = IpoptSolver()
        solver_id = solver.solver_id()
        prog.SetSolverOption(solver_id, "max_iter", int(self.ipopt_max_iter))
        prog.SetSolverOption(solver_id, "print_level", int(self.ipopt_print_level))
        prog.SetSolverOption(
            solver_id,
            "hessian_approximation",
            self.ipopt_hessian_approximation,
        )

        result = solver.Solve(prog)
        x_val = result.GetSolution(x)
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

        best_x = None
        best_metrics = None
        best_source = None
        if (
            callback_state["best_params"] is not None
            and callback_state["best_metrics"] is not None
        ):
            best_x = callback_state["best_params"].copy()
            best_metrics = dict(callback_state["best_metrics"])
            best_source = f"iteration_{callback_state['best_callback_count']}"

        return {
            "x": x_val,
            "f": final_metrics["objective"],
            "solver_f": result.get_optimal_cost(),
            "stats": {
                "return_status": str(result.get_solution_result()),
                "is_success": result.is_success(),
            },
            "identifiable_columns": identifiable_columns,
            "collision_checker": self.collision_checker,
            "best_x": best_x,
            "best_metrics": best_metrics,
            "best_source": best_source,
        }
