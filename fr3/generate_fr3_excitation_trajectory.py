import argparse
import copy
import sys
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_FR3_GRIPPER_URDF = (
    SCRIPT_DIR / "robot_description" / "fr3_description" / "fr3_gripper.urdf"
)
DEFAULT_CAMERA_BOX_XY_SCALE = 1.1
DEFAULT_CAMERA_BOX_Z_SCALE = 1.1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Franka FR3 excitation trajectory with Drake/IPOPT, "
            "workspace wall constraints, and camera-box obstacles."
        )
    )
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument(
        "--robot_urdf_path",
        type=Path,
        default=DEFAULT_FR3_GRIPPER_URDF,
        help=(
            "URDF used by the Drake collision checker. Defaults to the local "
            "FR3 gripper URDF; the optimized trajectory is still the 7-DOF arm."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("logs/fr3_excitation"),
        help="Directory where a timestamped trajectory run directory is created.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_initial_attempts", type=int, default=5000)
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=float, default=10.0)
    parser.add_argument(
        "--excite_type",
        type=str,
        choices=("cond", "condFriction"),
        default="cond",
    )
    parser.add_argument(
        "--friction_model",
        type=str,
        choices=("symmetric", "asymmetric"),
        default="symmetric",
    )
    parser.add_argument("--objective_lambda", type=float, default=1e-6)
    parser.add_argument("--eig_eps", type=float, default=1e-6)
    parser.add_argument(
        "--fourier_velocity_limit_scale",
        type=float,
        default=0.9,
        help="Velocity limit scale used by the IPOPT Fourier constraints.",
    )
    parser.add_argument(
        "--fourier_position_limit_scale",
        type=float,
        default=0.92,
        help="Position limit scale used by the IPOPT Fourier constraints.",
    )
    parser.add_argument("--ipopt_max_iter", type=int, default=1000)
    parser.add_argument("--ipopt_print_level", type=int, default=5)
    parser.add_argument(
        "--ipopt_hessian_approximation",
        type=str,
        choices=("limited-memory", "exact"),
        default="limited-memory",
    )
    parser.add_argument("--log_condition_every", type=int, default=5)
    parser.add_argument("--best_condition_check_every", type=int, default=3)
    parser.add_argument("--initial_best_condition_number", type=float, default=1e7)
    parser.add_argument(
        "--best_condition_fourier_velocity_margin_tolerance",
        type=float,
        default=0.0,
        help=(
            "Backward-compatible fallback for the best-condition velocity "
            "margin gate when no explicit margin minimum is passed to the solver."
        ),
    )
    parser.add_argument(
        "--best_condition_fourier_position_margin_tolerance",
        type=float,
        default=0.0,
        help=(
            "Backward-compatible fallback for the best-condition position "
            "margin gate when no explicit margin minimum is passed to the solver."
        ),
    )
    parser.add_argument(
        "--best_condition_fourier_velocity_margin_min",
        type=float,
        default=-0.01,
        help=(
            "Run the expensive best-candidate validation only when the IPOPT "
            "Fourier velocity margin is at least this value."
        ),
    )
    parser.add_argument(
        "--best_condition_fourier_position_margin_min",
        type=float,
        default=-0.001,
        help=(
            "Run the expensive best-candidate validation only when the IPOPT "
            "Fourier position margin is at least this value."
        ),
    )
    parser.add_argument("--early_stop_constraint_tol", type=float, default=1e-6)
    parser.add_argument("--camera_collision_stride", type=int, default=3)
    parser.add_argument("--validation_stride", type=int, default=1)
    parser.add_argument("--drake_min_distance", type=float, default=0.02)
    parser.add_argument("--drake_robot_sphere_radius", type=float, default=0.05)
    parser.add_argument("--drake_robot_link_samples", type=int, default=5)
    parser.add_argument("--drake_camera_chamfer_radius", type=float, default=0.015)
    parser.add_argument(
        "--drake_xy_prism_height",
        type=float,
        default=None,
        help=(
            "Optional full z height for XY-prism camera obstacles. Defaults to "
            "the physical camera-box height from the vendored FR3 setup."
        ),
    )
    parser.add_argument(
        "--drake_physical_camera_height",
        action="store_true",
        help="Use the physical camera-box z height even if drake_xy_prism_height is set.",
    )
    parser.add_argument(
        "--camera_box_xy_scale",
        type=float,
        default=DEFAULT_CAMERA_BOX_XY_SCALE,
        help="Scale applied to camera obstacle x/y dimensions.",
    )
    parser.add_argument(
        "--camera_box_z_scale",
        type=float,
        default=DEFAULT_CAMERA_BOX_Z_SCALE,
        help="Scale applied to camera obstacle z dimension.",
    )
    parser.add_argument("--link_y_lower", type=float, default=-0.4)
    parser.add_argument("--link_y_upper", type=float, default=0.4)
    parser.add_argument("--link_x_lower", type=float, default=-0.2)
    parser.add_argument("--link_z_lower", type=float, default=0.1)
    parser.add_argument(
        "--disable_self_collision_constraints",
        action="store_true",
        help="Disable the configured Franka self-collision sphere-pair constraints.",
    )
    parser.add_argument(
        "--self_collision_clearance",
        type=float,
        default=0.0,
        help="Minimum clearance for the configured Franka self-collision sphere pairs.",
    )
    parser.add_argument(
        "--disable_link_y_bounds",
        "--disable_wall_bounds",
        dest="disable_link_y_bounds",
        action="store_true",
        help="Disable the sampled wall bounds on robot links.",
    )
    parser.add_argument("--no_identifiable_column_reduction", action="store_true")
    parser.add_argument(
        "--initial_only",
        action="store_true",
        help="Stop after finding and saving a valid initial trajectory.",
    )
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def make_wall_bound_solver(base_solver_cls, flat_params_to_traj):
    class WallBoundDrakeMathematicalProgramExcitationSolver(base_solver_cls):
        def __init__(
            self,
            *args,
            link_x_lower=-0.2,
            link_z_lower=0.1,
            use_self_collision_constraints=True,
            self_collision_body_pairs=(),
            self_collision_clearance=0.0,
            **kwargs,
        ):
            self.link_x_lower = float(link_x_lower)
            self.link_z_lower = float(link_z_lower)
            self.use_self_collision_constraints = bool(use_self_collision_constraints)
            self.self_collision_body_pairs = tuple(self_collision_body_pairs)
            self.self_collision_clearance = float(self_collision_clearance)
            super().__init__(*args, **kwargs)

        def _path_constraint_values(self, flat_params):
            flat_params = np.asarray(flat_params, dtype=float).reshape(-1)
            _, q, _, _ = flat_params_to_traj(
                flat_params,
                self.fourier_config,
                self.robot_config,
            )
            q_sampled = q[:: self.camera_collision_stride]
            values = [
                self.collision_checker.minimum_distance_constraint_values(q_sampled)
            ]
            if self.use_self_collision_constraints:
                self_collision_margins = (
                    self.collision_checker.robot_self_collision_pair_margins(
                        q_sampled,
                        body_pairs=self.self_collision_body_pairs,
                    )
                )
                values.append(self_collision_margins.reshape(-1))
            if self.use_link_y_bounds:
                wall_margins = robot_link_workspace_margins(
                    self.collision_checker,
                    q_sampled,
                    x_lower=self.link_x_lower,
                    y_lower=self.link_y_lower,
                    y_upper=self.link_y_upper,
                    z_lower=self.link_z_lower,
                )
                values.append(wall_margins.reshape(-1))
            return np.concatenate(values)

        def _path_constraint_bounds(self):
            n_samples = len(
                range(
                    0,
                    int(self.fourier_config["duration"] * 100) + 1,
                    self.camera_collision_stride,
                )
            )
            n_self_collision_outputs = (
                len(self.self_collision_body_pairs)
                if self.use_self_collision_constraints
                else 0
            )
            n_wall_outputs = 4 if self.use_link_y_bounds else 0
            n_outputs = n_samples * (1 + n_self_collision_outputs + n_wall_outputs)
            lower = np.zeros(n_outputs, dtype=float)
            upper = np.full(n_outputs, np.inf, dtype=float)
            upper[:n_samples] = 1.0
            if n_self_collision_outputs:
                start = n_samples
                stop = start + n_samples * n_self_collision_outputs
                lower[start:stop] = self.self_collision_clearance
            return lower, upper

    return WallBoundDrakeMathematicalProgramExcitationSolver


def max_constraint_violation(values, lower, upper):
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    lower_violation = np.maximum(lower - values, 0.0)
    upper_violation = np.maximum(values - upper, 0.0)
    return float(max(np.max(lower_violation), np.max(upper_violation)))


def min_or_inf(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.inf
    return float(np.min(values))


def robot_link_workspace_margins(
    collision_checker,
    q,
    *,
    x_lower,
    y_lower,
    y_upper,
    z_lower,
):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    margins = []
    for q_i in q:
        xy_points, xy_radii = robot_workspace_sample_points_and_radii(
            collision_checker,
            q_i,
            min_link_index=1,
        )
        z_points, z_radii = robot_workspace_sample_points_and_radii(
            collision_checker,
            q_i,
            min_link_index=2,
        )
        margins.append(
            (
                float(np.min(xy_points[:, 0] - xy_radii)) - float(x_lower),
                float(np.min(xy_points[:, 1] - xy_radii)) - float(y_lower),
                float(y_upper) - float(np.max(xy_points[:, 1] + xy_radii)),
                float(np.min(z_points[:, 2] - z_radii)) - float(z_lower),
            )
        )
    return np.asarray(margins, dtype=float)


def robot_workspace_sample_points(collision_checker, q_i, *, min_link_index):
    points, _radii = robot_workspace_sample_points_and_radii(
        collision_checker,
        q_i,
        min_link_index=min_link_index,
    )
    return points


def robot_workspace_sample_points_and_radii(collision_checker, q_i, *, min_link_index):
    if not hasattr(collision_checker, "_robot_sample_specs"):
        points = collision_checker._robot_sample_points(q_i)
        radii = np.full(len(points), collision_checker.robot_sphere_radius)
        return points, radii

    collision_checker.plant.SetPositions(
        collision_checker.plant_context,
        collision_checker.model_instance,
        q_i,
    )
    points = []
    radii = []
    for body, offset, radius in collision_checker._robot_sample_specs:
        body_name = body.name()
        if not robot_body_at_or_after_link(body_name, min_link_index):
            continue
        x_wb = collision_checker.plant.EvalBodyPoseInWorld(
            collision_checker.plant_context,
            body,
        )
        points.append(x_wb.multiply(offset))
        radii.append(float(radius))
    if not points:
        points = collision_checker._robot_sample_points(q_i)
        radii = np.full(len(points), collision_checker.robot_sphere_radius)
        return points, radii
    return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)


def robot_body_at_or_after_link(body_name, min_link_index):
    if body_name in {"base", "world"}:
        return False
    if "link" in body_name:
        suffix = body_name.rsplit("link", 1)[-1]
        digits = "".join(char for char in suffix if char.isdigit())
        if digits:
            return int(digits) >= int(min_link_index)
    return int(min_link_index) <= 2


def scaled_camera_boxes(
    camera_box_cls,
    camera_box_specs_mm,
    *,
    xy_prism_height,
    xy_scale,
    z_scale,
):
    boxes = []
    for name, center_mm, size_mm in camera_box_specs_mm:
        center = np.asarray(center_mm, dtype=float) / 1000.0
        size = np.asarray(size_mm, dtype=float) / 1000.0
        size[:2] *= float(xy_scale)
        if xy_prism_height is None:
            size[2] *= float(z_scale)
        else:
            size[2] = float(xy_prism_height) * float(z_scale)
        boxes.append(camera_box_cls(name=name, center=center, size=size))
    return boxes


def initial_solver_constraint_report(
    flat_params,
    q,
    fourier_config,
    robot_config,
    collision_checker,
    args,
    fourier_constraint_bounds,
    fourier_constraint_values,
):
    fourier_lbg, fourier_ubg = fourier_constraint_bounds(
        fourier_config,
        robot_config,
    )
    fourier_values = fourier_constraint_values(
        flat_params,
        fourier_config,
        robot_config,
    )
    fourier_violation = max_constraint_violation(
        fourier_values,
        fourier_lbg,
        fourier_ubg,
    )

    q_sampled = q[:: max(1, int(args.camera_collision_stride))]
    path_values = [collision_checker.minimum_distance_constraint_values(q_sampled)]
    path_lbg = [np.zeros_like(path_values[0], dtype=float)]
    path_ubg = [np.ones_like(path_values[0], dtype=float)]
    self_collision_values = np.array([], dtype=float)
    if not args.disable_self_collision_constraints:
        self_collision_values = collision_checker.robot_self_collision_pair_margins(
            q_sampled
        ).reshape(-1)
        path_values.append(self_collision_values)
        path_lbg.append(
            np.full_like(
                self_collision_values,
                float(args.self_collision_clearance),
                dtype=float,
            )
        )
        path_ubg.append(np.full_like(self_collision_values, np.inf, dtype=float))
    if not args.disable_link_y_bounds:
        wall_values = robot_link_workspace_margins(
            collision_checker,
            q_sampled,
            x_lower=args.link_x_lower,
            y_lower=args.link_y_lower,
            y_upper=args.link_y_upper,
            z_lower=args.link_z_lower,
        ).reshape(-1)
        path_values.append(wall_values)
        path_lbg.append(np.zeros_like(wall_values, dtype=float))
        path_ubg.append(np.full_like(wall_values, np.inf, dtype=float))

    path_values = np.concatenate(path_values)
    path_lbg = np.concatenate(path_lbg)
    path_ubg = np.concatenate(path_ubg)
    path_violation = max_constraint_violation(path_values, path_lbg, path_ubg)

    njoints = int(robot_config["njoints"])
    fourier_values_by_joint = fourier_values.reshape(njoints, 5)
    fourier_ubg_by_joint = fourier_ubg.reshape(njoints, 5)
    n_collision = len(q_sampled)
    collision_values = path_values[:n_collision]
    collision_upper = path_ubg[:n_collision]
    n_self_collision = self_collision_values.size
    wall_values = path_values[n_collision + n_self_collision :]

    max_violation = max(fourier_violation, path_violation)
    return {
        "valid": max_violation <= max(0.0, float(args.early_stop_constraint_tol)),
        "max_violation": max_violation,
        "fourier_equality_residual": float(
            np.max(np.abs(fourier_values_by_joint[:, :3]))
        ),
        "fourier_velocity_margin": min_or_inf(
            fourier_ubg_by_joint[:, 3] - fourier_values_by_joint[:, 3]
        ),
        "fourier_position_margin": min_or_inf(
            fourier_ubg_by_joint[:, 4] - fourier_values_by_joint[:, 4]
        ),
        "drake_collision_margin": min_or_inf(
            np.minimum(collision_values, collision_upper - collision_values)
        ),
        "self_collision_margin": min_or_inf(
            self_collision_values - float(args.self_collision_clearance)
        ),
        "wall_margin": min_or_inf(wall_values),
    }


def make_collision_checker(
    args,
    drake_camera_collision_checker,
    camera_box_cls,
    camera_box_specs_mm,
):
    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    camera_boxes = scaled_camera_boxes(
        camera_box_cls,
        camera_box_specs_mm,
        xy_prism_height=xy_prism_height,
        xy_scale=args.camera_box_xy_scale,
        z_scale=args.camera_box_z_scale,
    )
    return drake_camera_collision_checker(
        robot_name=args.robot,
        robot_urdf_path=args.robot_urdf_path,
        min_distance=args.drake_min_distance,
        robot_sphere_radius=args.drake_robot_sphere_radius,
        robot_link_samples=args.drake_robot_link_samples,
        camera_boxes=camera_boxes,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
    )


def generate_valid_initial_trajectory(
    args,
    fourier_config,
    robot_config,
    collision_checker,
    logger,
    flatten_fourier_params,
    generate_random_param,
    is_traj_valid,
    obtain_fourier_traj,
    validate_trajectory,
    fourier_constraint_bounds,
    fourier_constraint_values,
):
    for attempt in range(1, int(args.max_initial_attempts) + 1):
        params = generate_random_param(
            int(fourier_config["order"]),
            int(robot_config["njoints"]),
        )
        t, q, dq, ddq = obtain_fourier_traj(params, fourier_config, robot_config)
        if not is_traj_valid(q, dq, ddq, robot_config):
            continue

        flat_params = flatten_fourier_params(params)
        report = initial_solver_constraint_report(
            flat_params,
            q,
            fourier_config,
            robot_config,
            collision_checker,
            args,
            fourier_constraint_bounds,
            fourier_constraint_values,
        )
        if not report["valid"]:
            if attempt % 50 == 0:
                logger.info(
                    "Discarding initial candidate "
                    f"{attempt}: max_violation={report['max_violation']}, "
                    f"drake_collision_margin={report['drake_collision_margin']}, "
                    f"self_collision_margin={report['self_collision_margin']}, "
                    f"wall_margin={report['wall_margin']}"
                )
            continue

        eval_result = validate_trajectory(
            "Initial",
            flat_params,
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        if eval_result["valid"]:
            logger.info(f"Accepted initial trajectory after {attempt} attempt(s).")
            return flat_params, eval_result

    raise RuntimeError(
        "Could not find a valid initial FR3 trajectory after "
        f"{args.max_initial_attempts} attempts."
    )


def make_trajectory_record(source, flat_params, eval_result, metrics):
    eval_summary = {
        key: value
        for key, value in eval_result.items()
        if key not in {"t", "q", "dq", "ddq"}
    }
    return {
        "source": source,
        "flat_params": np.asarray(flat_params, dtype=float),
        "metrics": dict(metrics),
        "condition_number": float(metrics.get("condition_number", np.inf)),
        "valid": bool(eval_result["valid"]),
        "eval": eval_summary,
        "t": eval_result["t"],
        "q": eval_result["q"],
        "dq": eval_result["dq"],
        "ddq": eval_result["ddq"],
    }


def select_best_valid_record(records):
    valid_records = [
        record for record in records if record is not None and record["valid"]
    ]
    if not valid_records:
        return None
    return min(
        valid_records,
        key=lambda record: (
            not np.isfinite(record["condition_number"]),
            record["condition_number"],
        ),
    )


def save_traj_csv(save_dir, name, t, q, dq, ddq):
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{name}.csv"
    data = np.column_stack([t, q, dq, ddq])
    njoints = q.shape[1]
    header = (
        ["t"]
        + [f"q_{i}" for i in range(njoints)]
        + [f"dq_{i}" for i in range(njoints)]
        + [f"ddq_{i}" for i in range(njoints)]
    )
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    return csv_path


def condition_number_filename_suffix(condition_number):
    condition_number = float(condition_number)
    if not np.isfinite(condition_number):
        return "nonfinite"
    text = f"{condition_number:.6g}"
    return text.replace("-", "neg").replace("+", "").replace(".", "p")


def unique_csv_stem(save_dir, stem):
    save_dir = Path(save_dir)
    candidate = stem
    counter = 2
    while (save_dir / f"{candidate}.csv").exists():
        candidate = f"{stem}_{counter}"
        counter += 1
    return candidate


def save_robot_payload_fourier_format(
    save_dir,
    flat_params,
    fourier_config,
    robot_config,
    unflatten_fourier_params,
):
    params = unflatten_fourier_params(flat_params, fourier_config, robot_config)
    a_external, b_external = params[0], params[1]
    omega = 2.0 * np.pi / float(fourier_config["duration"])
    harmonic_ids = np.arange(1, int(fourier_config["order"]) + 1, dtype=float)

    a_values = (a_external / (omega * harmonic_ids[:, None])).T
    b_values = (-b_external / (omega * harmonic_ids[:, None])).T
    q0_values = np.asarray(robot_config["init_pos"], dtype=float)

    np.save(save_dir / "a_value.npy", a_values)
    np.save(save_dir / "b_value.npy", b_values)
    np.save(save_dir / "q0_value.npy", q0_values)
    np.save(save_dir / "omega.npy", np.array([omega]))
    np.save(save_dir / "external_flat_fourier_params.npy", flat_params)


def yaml_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: yaml_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(val) for val in value]
    return value


def main():
    args = parse_args()
    original_cwd = Path.cwd()
    save_root = args.save_dir.expanduser()
    if not save_root.is_absolute():
        save_root = (original_cwd / save_root).resolve()
    run_name = args.run_name or datetime.now().strftime("fr3_excitation_%Y%m%d_%H%M%S")
    experiment_dir = save_root / run_name

    from generate_excitation_ipopt_drake import (
        validate_trajectory as base_validate_trajectory,
    )
    from loguru import logger
    from system_identification.drake.camera_collision import (
        CAMERA_BOX_SPECS_MM,
        DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
        CameraBox,
        DrakeCameraCollisionChecker,
    )
    from system_identification.drake.mathematical_program_solver import (
        DrakeMathematicalProgramExcitationSolver,
        fourier_constraint_bounds,
        fourier_constraint_values,
    )
    from system_identification.excitation_generator_new import (
        generate_random_param,
        is_traj_valid,
        obtain_fourier_traj,
    )
    from system_identification.fourier_utils import (
        flat_params_to_traj,
        flatten_fourier_params,
        unflatten_fourier_params,
    )
    from system_identification.inertia_model import InertiaModel
    from system_identification.ipopt_solver import evaluate_params_metrics
    from system_identification.utils import retrieve_robot_config, vis_compare_seqs

    def optimization_fourier_constraint_bounds(fourier_config, robot_config):
        return fourier_constraint_bounds(
            fourier_config,
            robot_config,
            velocity_limit_scale=args.fourier_velocity_limit_scale,
            position_limit_scale=args.fourier_position_limit_scale,
        )

    def validate_trajectory(
        label,
        flat_params,
        fourier_config,
        robot_config,
        collision_checker,
        args,
    ):
        base_args = copy.copy(args)
        base_args.disable_link_y_bounds = True
        result = base_validate_trajectory(
            label,
            flat_params,
            fourier_config,
            robot_config,
            collision_checker,
            base_args,
        )
        validation_stride = max(1, int(args.validation_stride))
        q_sampled = result["q"][::validation_stride]
        workspace_margin = np.inf
        workspace_report = {}
        if not args.disable_link_y_bounds:
            workspace_margins = robot_link_workspace_margins(
                collision_checker,
                q_sampled,
                x_lower=args.link_x_lower,
                y_lower=args.link_y_lower,
                y_upper=args.link_y_upper,
                z_lower=args.link_z_lower,
            )
            workspace_margin = float(np.min(workspace_margins))
            workspace_report = {
                "link_wall_margin": workspace_margin,
                "min_x_lower_margin": float(np.min(workspace_margins[:, 0])),
                "min_y_lower_margin": float(np.min(workspace_margins[:, 1])),
                "min_y_upper_margin": float(np.min(workspace_margins[:, 2])),
                "min_z_margin": float(np.min(workspace_margins[:, 3])),
            }
        self_collision_margin = np.inf
        if not args.disable_self_collision_constraints:
            self_collision_margin = float(
                np.min(
                    collision_checker.robot_self_collision_pair_margins(q_sampled)
                    - float(args.self_collision_clearance)
                )
            )
        result.update(
            {
                "valid": bool(
                    result["valid"]
                    and workspace_margin >= 0.0
                    and self_collision_margin >= 0.0
                ),
                "self_collision_margin": self_collision_margin,
                **workspace_report,
            }
        )
        if not args.disable_link_y_bounds:
            logger.info(
                f"{label} sampled workspace margins: "
                f"x_lower={result['min_x_lower_margin']}; "
                f"y_lower={result['min_y_lower_margin']}; "
                f"y_upper={result['min_y_upper_margin']}; "
                f"z_lower={result['min_z_margin']}"
            )
        if not args.disable_self_collision_constraints:
            logger.info(
                f"{label} sampled self-collision margin: "
                f"{result['self_collision_margin']}"
            )
        return result

    if not args.no_save:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "args.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_safe(vars(args)), f, sort_keys=True)

    np.random.seed(args.seed)

    logger.info(f"Using local FR3 helpers and models under {SCRIPT_DIR}")
    logger.info(f"Using Drake collision URDF: {args.robot_urdf_path}")
    logger.info(
        "FR3 workspace constraints: "
        f"x > {args.link_x_lower}, "
        f"{args.link_y_lower} < y < {args.link_y_upper}, "
        f"z > {args.link_z_lower}; camera min distance {args.drake_min_distance}"
    )
    logger.info(
        "FR3 camera boxes: x/y scale %s, z scale %s",
        args.camera_box_xy_scale,
        args.camera_box_z_scale,
    )
    logger.info(
        "IPOPT Fourier constraint scales: velocity=%s, position=%s; "
        "validation remains at 0.95x",
        args.fourier_velocity_limit_scale,
        args.fourier_position_limit_scale,
    )
    logger.info(
        "Best-condition validation margin thresholds: velocity>=%s, position>=%s",
        args.best_condition_fourier_velocity_margin_min,
        args.best_condition_fourier_position_margin_min,
    )
    if not args.disable_self_collision_constraints:
        logger.info(
            "FR3 self-collision constraints: %d configured body pairs, "
            "clearance >= %s",
            len(DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS),
            args.self_collision_clearance,
        )
    logger.info(f"Output directory: {experiment_dir}")

    robot_config = retrieve_robot_config(args.robot)
    fourier_config = {
        "order": int(args.fourier_order),
        "duration": float(args.fourier_duration),
    }
    collision_checker = make_collision_checker(
        args,
        DrakeCameraCollisionChecker,
        CameraBox,
        CAMERA_BOX_SPECS_MM,
    )

    init_params, initial_eval = generate_valid_initial_trajectory(
        args,
        fourier_config,
        robot_config,
        collision_checker,
        logger,
        flatten_fourier_params,
        generate_random_param,
        is_traj_valid,
        obtain_fourier_traj,
        validate_trajectory,
        optimization_fourier_constraint_bounds,
        fourier_constraint_values,
    )
    if not args.no_save:
        save_traj_csv(
            experiment_dir,
            "initial",
            initial_eval["t"],
            initial_eval["q"],
            initial_eval["dq"],
            initial_eval["ddq"],
        )

    if args.plot:
        vis_compare_seqs(
            [initial_eval["t"], initial_eval["t"], initial_eval["t"]],
            [initial_eval["q"], initial_eval["dq"], initial_eval["ddq"]],
            ["q", "dq", "ddq"],
            ["time"],
        )

    if args.initial_only:
        if not args.no_save:
            save_traj_csv(
                experiment_dir,
                "selected",
                initial_eval["t"],
                initial_eval["q"],
                initial_eval["dq"],
                initial_eval["ddq"],
            )
            save_robot_payload_fourier_format(
                experiment_dir,
                init_params,
                fourier_config,
                robot_config,
                unflatten_fourier_params,
            )
            summary = {
                "selected_source": "initial",
                "selected_eval": initial_eval,
                "ipopt_stats": None,
            }
            with open(experiment_dir / "summary.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(yaml_safe(summary), f, sort_keys=True)
        logger.info(
            "Stopping after valid initial trajectory because --initial_only was set."
        )
        return

    best_callback_record = {"record": None}

    def save_best_valid_candidate(flat_params, metrics, iteration):
        eval_result = validate_trajectory(
            f"Iteration {iteration} best-condition candidate",
            flat_params,
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        if not eval_result["valid"]:
            logger.info(
                f"Iteration {iteration} improved condition number "
                f"{metrics['condition_number']}, but failed validation."
            )
            return False

        record = make_trajectory_record(
            "best_condition",
            flat_params,
            eval_result,
            metrics,
        )
        best_callback_record["record"] = record
        if not args.no_save:
            condition_suffix = condition_number_filename_suffix(
                metrics["condition_number"]
            )
            archive_name = unique_csv_stem(
                experiment_dir,
                f"best_condition_{condition_suffix}",
            )
            save_traj_csv(
                experiment_dir,
                archive_name,
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            save_traj_csv(
                experiment_dir,
                "best_condition",
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            logger.info(f"Saved valid best-condition archive: {archive_name}.csv")
        logger.info(
            f"Accepted valid best candidate at IPOPT iteration {iteration}: "
            f"condition_number={metrics['condition_number']}"
        )
        return True

    solver_cls = make_wall_bound_solver(
        DrakeMathematicalProgramExcitationSolver,
        flat_params_to_traj,
    )
    inertia_model = InertiaModel(args.robot)
    solver = solver_cls(
        fourier_config=fourier_config,
        robot_config=robot_config,
        inertia_model=inertia_model,
        robot_name=args.robot,
        friction_model=args.friction_model
        if args.excite_type == "condFriction"
        else None,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
        camera_collision_stride=args.camera_collision_stride,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_print_level=args.ipopt_print_level,
        ipopt_hessian_approximation=args.ipopt_hessian_approximation,
        log_condition_every=args.log_condition_every,
        early_stop_constraint_tol=args.early_stop_constraint_tol,
        use_identifiable_columns=not args.no_identifiable_column_reduction,
        link_x_lower=args.link_x_lower,
        link_y_lower=args.link_y_lower,
        link_y_upper=args.link_y_upper,
        link_z_lower=args.link_z_lower,
        use_link_y_bounds=not args.disable_link_y_bounds,
        use_self_collision_constraints=not args.disable_self_collision_constraints,
        self_collision_body_pairs=DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
        self_collision_clearance=args.self_collision_clearance,
        best_condition_initial=args.initial_best_condition_number,
        best_candidate_check_every=args.best_condition_check_every,
        best_candidate_fourier_velocity_margin_tolerance=(
            args.best_condition_fourier_velocity_margin_tolerance
        ),
        best_candidate_fourier_position_margin_tolerance=(
            args.best_condition_fourier_position_margin_tolerance
        ),
        best_candidate_fourier_velocity_margin_min=(
            args.best_condition_fourier_velocity_margin_min
        ),
        best_candidate_fourier_position_margin_min=(
            args.best_condition_fourier_position_margin_min
        ),
        fourier_velocity_limit_scale=args.fourier_velocity_limit_scale,
        fourier_position_limit_scale=args.fourier_position_limit_scale,
        best_candidate_callback=save_best_valid_candidate,
        collision_checker=collision_checker,
    )

    start = time.perf_counter()
    result = solver.solve(init_params)
    elapsed = time.perf_counter() - start
    stats = result["stats"]
    logger.info(f"Drake MathematicalProgram IPOPT time cost: {elapsed}")
    logger.info(f"Drake MathematicalProgram IPOPT status: {stats['return_status']}")
    logger.info(f"Drake MathematicalProgram IPOPT success: {stats['is_success']}")
    logger.info(f"Recomputed objective after optimization: {result['f']}")
    logger.info(f"Solver-reported objective: {result['solver_f']}")

    identifiable_columns = result["identifiable_columns"]
    initial_metrics = evaluate_params_metrics(
        init_params,
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=args.friction_model
        if args.excite_type == "condFriction"
        else None,
        identifiable_columns=identifiable_columns,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
    )
    initial_record = make_trajectory_record(
        "initial",
        init_params,
        initial_eval,
        initial_metrics,
    )

    final_metrics = evaluate_params_metrics(
        result["x"],
        fourier_config,
        robot_config,
        inertia_model,
        friction_model=args.friction_model
        if args.excite_type == "condFriction"
        else None,
        identifiable_columns=identifiable_columns,
        eig_eps=args.eig_eps,
        min_eig_weight=args.objective_lambda,
    )
    final_eval = validate_trajectory(
        "Final",
        result["x"],
        fourier_config,
        robot_config,
        collision_checker,
        args,
    )
    final_record = make_trajectory_record(
        "final", result["x"], final_eval, final_metrics
    )

    best_record = best_callback_record["record"]
    if result.get("best_x") is not None:
        best_eval = validate_trajectory(
            "Best-condition",
            result["best_x"],
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
        best_record = make_trajectory_record(
            "best_condition",
            result["best_x"],
            best_eval,
            result["best_metrics"],
        )

    selected_record = select_best_valid_record(
        [best_record, final_record, initial_record]
    )
    if selected_record is None:
        raise RuntimeError("No valid FR3 excitation trajectory was produced.")

    if not args.no_save:
        save_traj_csv(
            experiment_dir,
            "final",
            final_record["t"],
            final_record["q"],
            final_record["dq"],
            final_record["ddq"],
        )
        selected_csv = save_traj_csv(
            experiment_dir,
            "selected",
            selected_record["t"],
            selected_record["q"],
            selected_record["dq"],
            selected_record["ddq"],
        )
        save_robot_payload_fourier_format(
            experiment_dir,
            selected_record["flat_params"],
            fourier_config,
            robot_config,
            unflatten_fourier_params,
        )
        summary = {
            "selected_source": selected_record["source"],
            "selected_condition_number": selected_record["condition_number"],
            "selected_eval": selected_record["eval"],
            "selected_metrics": selected_record["metrics"],
            "ipopt_stats": stats,
            "elapsed_s": elapsed,
            "csv": str(selected_csv),
        }
        with open(experiment_dir / "summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_safe(summary), f, sort_keys=True)

    if args.plot:
        vis_compare_seqs(
            [selected_record["t"], selected_record["t"], selected_record["t"]],
            [selected_record["q"], selected_record["dq"], selected_record["ddq"]],
            ["q", "dq", "ddq"],
            ["time"],
        )

    logger.info(
        f"Selected {selected_record['source']} trajectory with condition number "
        f"{selected_record['condition_number']}"
    )
    if not args.no_save:
        logger.info(
            f"Saved selected trajectory and Fourier parameters to {experiment_dir}"
        )


if __name__ == "__main__":
    main()
