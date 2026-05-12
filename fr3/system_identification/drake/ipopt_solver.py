import casadi as ca
import numpy as np

from loguru import logger
from system_identification.drake.camera_collision import DrakeCameraCollisionChecker
from system_identification.fourier_utils import flat_params_to_traj
from system_identification.ipopt_solver import (
    ConditionNumberIterationCallback,
    ExcitationObjectiveCallback,
    build_fourier_constraints,
    build_regressor,
    evaluate_params_metrics,
    select_identifiable_columns,
)


class DrakeCollisionCallback(ca.Callback):
    def __init__(
        self,
        name,
        fourier_config,
        robot_config,
        collision_checker,
        stride=5,
    ):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._collision_checker = collision_checker
        self._stride = max(1, int(stride))
        self._n_params = 2 * fourier_config["order"] * robot_config["njoints"]
        self._n_samples = len(
            range(0, int(fourier_config["duration"] * 100) + 1, self._stride)
        )
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _idx):
        return ca.Sparsity.dense(self._n_params, 1)

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.dense(self._n_samples, 1)

    def eval(self, args):
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self._fourier_config,
            self._robot_config,
        )
        return [ca.DM(self._collision_checker.constraint_values(q[:: self._stride]))]


class DrakeLinkYBoundsCallback(ca.Callback):
    def __init__(
        self,
        name,
        fourier_config,
        robot_config,
        collision_checker,
        lower=-0.45,
        upper=0.35,
        stride=5,
    ):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._collision_checker = collision_checker
        self._lower = lower
        self._upper = upper
        self._stride = max(1, int(stride))
        self._n_params = 2 * fourier_config["order"] * robot_config["njoints"]
        self._n_samples = len(
            range(0, int(fourier_config["duration"] * 100) + 1, self._stride)
        )
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _idx):
        return ca.Sparsity.dense(self._n_params, 1)

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.dense(2 * self._n_samples, 1)

    def eval(self, args):
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self._fourier_config,
            self._robot_config,
        )
        margins = self._collision_checker.robot_link_y_margins(
            q[:: self._stride],
            lower=self._lower,
            upper=self._upper,
        )
        return [ca.DM(margins.reshape(-1))]


class DrakePathConstraintsCallback(ca.Callback):
    def __init__(
        self,
        name,
        fourier_config,
        robot_config,
        collision_checker,
        lower=-0.45,
        upper=0.35,
        z_lower=0.0,
        stride=20,
        use_link_y_bounds=True,
    ):
        ca.Callback.__init__(self)
        self._fourier_config = fourier_config
        self._robot_config = robot_config
        self._collision_checker = collision_checker
        self._lower = lower
        self._upper = upper
        self._z_lower = z_lower
        self._stride = max(1, int(stride))
        self._use_link_y_bounds = use_link_y_bounds
        self._n_params = 2 * fourier_config["order"] * robot_config["njoints"]
        self._n_samples = len(
            range(0, int(fourier_config["duration"] * 100) + 1, self._stride)
        )
        self._n_outputs = self._n_samples * (4 if use_link_y_bounds else 1)
        self.construct(name, {"enable_fd": True})

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, _idx):
        return ca.Sparsity.dense(self._n_params, 1)

    def get_sparsity_out(self, _idx):
        return ca.Sparsity.dense(self._n_outputs, 1)

    def eval(self, args):
        flat_params = np.array(args[0], dtype=float).reshape(-1)
        _, q, _, _ = flat_params_to_traj(
            flat_params,
            self._fourier_config,
            self._robot_config,
        )
        q_sampled = q[:: self._stride]
        values = [self._collision_checker.constraint_values(q_sampled)]
        if self._use_link_y_bounds:
            margins = self._collision_checker.robot_link_wall_margins(
                q_sampled,
                y_lower=self._lower,
                y_upper=self._upper,
                z_lower=self._z_lower,
            )
            values.append(margins.reshape(-1))
        return [ca.DM(np.concatenate(values))]


def build_fourier_constraints_with_drake_collision(
    x,
    fourier_config,
    robot_config,
    drake_collision_callback,
    link_y_bounds_callback=None,
):
    g, lbg, ubg = build_fourier_constraints(
        x,
        fourier_config,
        robot_config,
        camera_clearance_callback=None,
    )
    collision_g = drake_collision_callback(x)
    g_parts = [g, collision_g]
    lbg_parts = [lbg, np.zeros(int(collision_g.shape[0]))]
    ubg_parts = [ubg, np.full(int(collision_g.shape[0]), np.inf)]
    if link_y_bounds_callback is not None:
        link_y_g = link_y_bounds_callback(x)
        g_parts.append(link_y_g)
        lbg_parts.append(np.zeros(int(link_y_g.shape[0])))
        ubg_parts.append(np.full(int(link_y_g.shape[0]), np.inf))

    return (
        ca.vertcat(*g_parts),
        np.concatenate(lbg_parts),
        np.concatenate(ubg_parts),
    )


class IpoptExcitationDrakeSolver:
    def __init__(
        self,
        fourier_config,
        robot_config,
        inertia_model,
        robot_name="fr3",
        friction_model=None,
        eig_eps=1e-9,
        min_eig_weight=1e-6,
        camera_collision_stride=20,
        ipopt_max_iter=300,
        ipopt_print_level=5,
        ipopt_hessian_approximation="limited-memory",
        log_condition_every=5,
        early_stop_objective=None,
        early_stop_constraint_tol=1e-6,
        use_identifiable_columns=True,
        drake_min_distance=0.0,
        drake_robot_sphere_radius=0.05,
        drake_robot_link_samples=5,
        drake_camera_chamfer_radius=0.02,
        drake_xy_prism_height=None,
        link_y_lower=-0.45,
        link_y_upper=0.35,
        link_z_lower=0.0,
        use_link_y_bounds=True,
        best_condition_initial=1.0e7,
        best_candidate_check_every=3,
        best_candidate_callback=None,
        collision_checker=None,
    ):
        self.fourier_config = fourier_config
        self.robot_config = robot_config
        self.inertia_model = inertia_model
        self.robot_name = robot_name
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
        self.link_y_lower = link_y_lower
        self.link_y_upper = link_y_upper
        self.link_z_lower = link_z_lower
        self.use_link_y_bounds = use_link_y_bounds
        self.best_condition_initial = best_condition_initial
        self.best_candidate_check_every = best_candidate_check_every
        self.best_candidate_callback = best_candidate_callback
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

    def solve(self, initial_params):
        initial_params = np.asarray(initial_params, dtype=float).reshape(-1)
        n_params = initial_params.size
        x = ca.MX.sym("fourier_params", n_params)

        identifiable_columns = self._initial_identifiable_columns(initial_params)
        objective_callback = ExcitationObjectiveCallback(
            "excitation_objective_drake",
            self.fourier_config,
            self.robot_config,
            self.inertia_model,
            friction_model=self.friction_model,
            identifiable_columns=identifiable_columns,
            eig_eps=self.eig_eps,
            min_eig_weight=self.min_eig_weight,
        )
        drake_collision_callback = DrakePathConstraintsCallback(
            "drake_path_constraints",
            self.fourier_config,
            self.robot_config,
            self.collision_checker,
            stride=self.camera_collision_stride,
            lower=self.link_y_lower,
            upper=self.link_y_upper,
            z_lower=self.link_z_lower,
            use_link_y_bounds=self.use_link_y_bounds,
        )
        g, lbg, ubg = build_fourier_constraints_with_drake_collision(
            x,
            self.fourier_config,
            self.robot_config,
            drake_collision_callback,
        )
        nlp = {"x": x, "f": objective_callback(x), "g": g}
        solver_options = {
            "ipopt.max_iter": self.ipopt_max_iter,
            "ipopt.print_level": self.ipopt_print_level,
            "ipopt.hessian_approximation": self.ipopt_hessian_approximation,
            "print_time": False,
        }
        iteration_callback = ConditionNumberIterationCallback(
            "condition_number_iteration_drake",
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
            log_metrics=bool(self.log_condition_every and self.log_condition_every > 0),
            log_every=max(1, self.log_condition_every),
            best_condition_initial=self.best_condition_initial,
            best_candidate_check_every=self.best_candidate_check_every,
            best_candidate_callback=self.best_candidate_callback,
        )
        solver_options["iteration_callback"] = iteration_callback
        solver_options["iteration_callback_step"] = 1

        solver = ca.nlpsol("excitation_ipopt_drake", "ipopt", nlp, solver_options)
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
        best_x = None
        best_metrics = None
        best_source = None
        if (
            iteration_callback.best_params is not None
            and iteration_callback.best_metrics is not None
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
            "collision_checker": self.collision_checker,
            "best_x": best_x,
            "best_metrics": best_metrics,
            "best_source": best_source,
            "stopped_early": bool(
                iteration_callback is not None and iteration_callback.stopped_early
            ),
        }
