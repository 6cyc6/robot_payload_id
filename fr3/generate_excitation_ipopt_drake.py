import argparse
import os

from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from loguru import logger
from system_identification.drake.camera_collision import DrakeCameraCollisionChecker
from system_identification.drake.ipopt_solver import IpoptExcitationDrakeSolver
from system_identification.excitation_generator_new import (
    is_traj_valid,
    obtain_valid_traj_param,
)
from system_identification.fourier_utils import (
    flat_params_to_traj,
    flatten_fourier_params,
    unflatten_fourier_params,
)
from system_identification.inertia_model import InertiaModel
from system_identification.ipopt_solver import ipopt_status_is_acceptable
from system_identification.my_utils.path_utils import safe_mkdir
from system_identification.utils import retrieve_robot_config, vis_compare_seqs


def save_traj_csv(save_dir, name, t, q, dq, ddq):
    safe_mkdir(save_dir)
    csv_path = Path(save_dir) / f"{name}.csv"
    data = np.column_stack([t, q, dq, ddq])
    njoints = q.shape[1]
    header = (
        ["t"]
        + [f"q_{i}" for i in range(njoints)]
        + [f"dq_{i}" for i in range(njoints)]
        + [f"ddq_{i}" for i in range(njoints)]
    )
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments="")
    logger.info(f"Saved trajectory CSV to {csv_path}")
    return csv_path


def fourier_constraint_report(flat_params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    omega = 2.0 * np.pi / duration
    harmonic_ids = np.arange(1, order + 1, dtype=float)
    params = unflatten_fourier_params(flat_params, fourier_config, robot_config)
    a_params, b_params = params[0], params[1]

    equality_residual = max(
        float(np.max(np.abs(np.sum(a_params, axis=0)))),
        float(np.max(np.abs(np.sum(b_params / harmonic_ids[:, None], axis=0)))),
        float(np.max(np.abs(np.sum(b_params * harmonic_ids[:, None], axis=0)))),
    )

    velocity_limit = np.minimum(
        np.array(robot_config["joint_vel_limits"], dtype=float),
        10000.0,
    )
    upper_pos = np.array(robot_config["upper_joint_pos_limits"], dtype=float)
    lower_pos = np.array(robot_config["lower_joint_pos_limits"], dtype=float)
    if len(upper_pos) > 1:
        lower_pos[1] = -1.0
        upper_pos[1] = 1.0
    lower_pos *= 0.95
    upper_pos *= 0.95
    init_pos = np.array(robot_config["init_pos"], dtype=float)
    offset_limit = np.minimum(upper_pos - init_pos, init_pos - lower_pos)

    root = np.sqrt(a_params**2 + b_params**2)
    velocity_margin = float(np.min(0.95 * velocity_limit - np.sum(root, axis=0)))
    position_margin = float(
        np.min(offset_limit * omega - np.sum(root / harmonic_ids[:, None], axis=0))
    )
    return {
        "equality_residual": equality_residual,
        "velocity_margin": velocity_margin,
        "position_margin": position_margin,
    }


def validate_trajectory(
    label,
    flat_params,
    fourier_config,
    robot_config,
    collision_checker,
    args,
):
    t, q, dq, ddq = flat_params_to_traj(
        flat_params,
        fourier_config,
        robot_config,
    )
    fourier_report = fourier_constraint_report(
        flat_params, fourier_config, robot_config
    )
    equality_tol = max(0.0, float(args.early_stop_constraint_tol))
    valid_joint_limits = bool(is_traj_valid(q, dq, ddq, robot_config))
    validation_stride = max(1, int(args.validation_stride))
    q_sampled = q[::validation_stride]
    collision_values = collision_checker.constraint_values(q_sampled)
    drake_collision_margin = float(np.min(collision_values))
    drake_clearance = float(collision_checker.min_clearance(q_sampled))
    link_wall_margin = np.inf
    if not args.disable_link_y_bounds:
        link_wall_margin = float(
            np.min(
                collision_checker.robot_link_wall_margins(
                    q_sampled,
                    y_lower=args.link_y_lower,
                    y_upper=args.link_y_upper,
                    z_lower=args.link_z_lower,
                )
            )
        )
    valid_fourier_constraints = (
        fourier_report["equality_residual"] <= equality_tol
        and fourier_report["velocity_margin"] >= 0.0
        and fourier_report["position_margin"] >= 0.0
    )
    valid = (
        valid_fourier_constraints
        and valid_joint_limits
        and drake_collision_margin >= 0.0
        and link_wall_margin >= 0.0
    )
    logger.info(
        f"{label} trajectory validity with Drake camera boxes: {valid}; "
        f"Fourier constraints valid: {valid_fourier_constraints}; "
        f"Fourier equality residual: {fourier_report['equality_residual']}; "
        f"Fourier velocity margin: {fourier_report['velocity_margin']}; "
        f"Fourier position margin: {fourier_report['position_margin']}; "
        f"joint limits valid: {valid_joint_limits}; "
        f"minimum Drake collision margin: {drake_collision_margin}; "
        f"minimum signed clearance: {drake_clearance}; "
        f"minimum link wall margin: {link_wall_margin}; "
        f"validation stride: {validation_stride}"
    )
    return {
        "t": t,
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "valid": valid,
        "valid_fourier_constraints": valid_fourier_constraints,
        "valid_joint_limits": valid_joint_limits,
        **fourier_report,
        "drake_collision_margin": drake_collision_margin,
        "drake_clearance": drake_clearance,
        "link_wall_margin": link_wall_margin,
    }


def main():
    os.chdir(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser(
        description="IPOPT excitation trajectory optimizer with Drake collision constraints"
    )
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--fourier_duration", type=int, default=10)
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
    parser.add_argument("--eig_eps", type=float, default=1e-9)
    parser.add_argument("--ipopt_max_iter", type=int, default=300)
    parser.add_argument("--ipopt_print_level", type=int, default=5)
    parser.add_argument(
        "--ipopt_hessian_approximation",
        type=str,
        choices=("limited-memory", "exact"),
        default="limited-memory",
    )
    parser.add_argument("--log_condition_every", type=int, default=5)
    parser.add_argument("--best_condition_check_every", type=int, default=3)
    parser.add_argument(
        "--initial_best_condition_number", type=float, default=10000000.0
    )
    parser.add_argument("--early_stop_objective", type=float, default=None)
    parser.add_argument("--early_stop_constraint_tol", type=float, default=1e-6)
    parser.add_argument("--camera_collision_stride", type=int, default=3)
    parser.add_argument("--validation_stride", type=int, default=1)
    parser.add_argument("--drake_min_distance", type=float, default=0.02)
    parser.add_argument("--drake_robot_sphere_radius", type=float, default=0.05)
    parser.add_argument("--drake_robot_link_samples", type=int, default=5)
    parser.add_argument("--drake_camera_chamfer_radius", type=float, default=0.0)
    parser.add_argument("--link_y_lower", type=float, default=-0.45)
    parser.add_argument("--link_y_upper", type=float, default=0.35)
    parser.add_argument("--link_z_lower", type=float, default=0.0)
    parser.add_argument("--disable_link_y_bounds", action="store_true")
    parser.add_argument(
        "--drake_xy_prism_height",
        type=float,
        default=None,
        help="Optional full z height for XY-prism camera obstacles. Defaults to physical camera-box height.",
    )
    parser.add_argument("--drake_physical_camera_height", action="store_true")
    parser.add_argument("--no_identifiable_column_reduction", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saves")
    args = parser.parse_args()

    experiment_dir = None
    if not args.no_save:
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        experiment_dir = (
            Path(args.save_dir)
            / f"{args.excite_type}_ipopt_drake_{args.robot}_{timestamp}"
        )
        safe_mkdir(experiment_dir)
        logger.info(f"Saving this experiment under {experiment_dir}")

    robot_config = retrieve_robot_config(args.robot)
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}
    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    collision_checker = DrakeCameraCollisionChecker(
        robot_name=args.robot,
        min_distance=args.drake_min_distance,
        robot_sphere_radius=args.drake_robot_sphere_radius,
        robot_link_samples=args.drake_robot_link_samples,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
    )

    while True:
        t, qs, qds, qdds, init_params = obtain_valid_traj_param(
            fourier_config,
            robot_config,
        )
        init_collision_margin = np.min(
            collision_checker.constraint_values(qs[:: args.camera_collision_stride])
        )
        init_link_wall_margin = np.inf
        if not args.disable_link_y_bounds:
            init_link_wall_margin = float(
                np.min(
                    collision_checker.robot_link_wall_margins(
                        qs[:: args.camera_collision_stride],
                        y_lower=args.link_y_lower,
                        y_upper=args.link_y_upper,
                        z_lower=args.link_z_lower,
                    )
                )
            )
        if init_collision_margin >= 0.0 and init_link_wall_margin >= 0.0:
            break
        logger.info(
            f"Discarding initial trajectory with Drake collision margin "
            f"{init_collision_margin} and link wall margin {init_link_wall_margin}"
        )

    init_params_flat = flatten_fourier_params(init_params)
    initial_eval = validate_trajectory(
        "Initial",
        init_params_flat,
        fourier_config,
        robot_config,
        collision_checker,
        args,
    )
    if experiment_dir is not None and initial_eval["valid"]:
        save_traj_csv(
            experiment_dir,
            "initial",
            initial_eval["t"],
            initial_eval["q"],
            initial_eval["dq"],
            initial_eval["ddq"],
        )

    if not args.no_plot:
        vis_compare_seqs([t, t, t], [qs, qds, qdds], ["q", "qd", "qdd"], ["time"])
        plt.show()

    def save_best_valid_candidate(flat_params, metrics, iteration):
        condition_number = float(metrics["condition_number"])
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
                f"Iteration {iteration} condition number {condition_number} beats the "
                "current best threshold, but the trajectory is invalid; not saving."
            )
            return False
        if experiment_dir is not None:
            save_traj_csv(
                experiment_dir,
                "best_condition",
                eval_result["t"],
                eval_result["q"],
                eval_result["dq"],
                eval_result["ddq"],
            )
            logger.info(
                f"Saved new valid best trajectory at IPOPT iteration {iteration}: "
                f"condition_number={condition_number}"
            )
        else:
            logger.info(
                f"Accepted new valid best trajectory at IPOPT iteration {iteration}: "
                f"condition_number={condition_number}"
            )
        return True

    inertia_model = InertiaModel(args.robot)
    solver = IpoptExcitationDrakeSolver(
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
        early_stop_objective=args.early_stop_objective,
        early_stop_constraint_tol=args.early_stop_constraint_tol,
        use_identifiable_columns=not args.no_identifiable_column_reduction,
        link_y_lower=args.link_y_lower,
        link_y_upper=args.link_y_upper,
        link_z_lower=args.link_z_lower,
        use_link_y_bounds=not args.disable_link_y_bounds,
        best_condition_initial=args.initial_best_condition_number,
        best_candidate_check_every=args.best_condition_check_every,
        best_candidate_callback=save_best_valid_candidate,
        collision_checker=collision_checker,
    )

    time1 = perf_counter()
    result = solver.solve(init_params_flat)
    time2 = perf_counter()
    ipopt_status = result["stats"].get("return_status")
    logger.info(f"IPOPT Drake time cost: {time2 - time1}")
    logger.info(f"IPOPT Drake recomputed objective after optimization: {result['f']}")
    logger.info(f"IPOPT Drake solver-reported objective: {result['solver_f']}")
    logger.info(f"IPOPT Drake return status: {ipopt_status}")
    logger.info(
        f"IPOPT Drake stopped by early-stop criterion: {result['stopped_early']}"
    )
    best_metrics = result.get("best_metrics")
    if best_metrics is None:
        logger.info(
            "No checked valid trajectory improved below the initial best condition "
            f"number {args.initial_best_condition_number}."
        )
    else:
        logger.info(
            f"Lowest valid checked candidate source: {result.get('best_source')}; "
            f"condition_number: {best_metrics.get('condition_number')}; "
            f"objective: {best_metrics.get('objective')}"
        )

    final_eval = validate_trajectory(
        "Final",
        result["x"],
        fourier_config,
        robot_config,
        collision_checker,
        args,
    )
    if result.get("best_x") is not None:
        validate_trajectory(
            "Best-condition",
            result["best_x"],
            fourier_config,
            robot_config,
            collision_checker,
            args,
        )
    ipopt_status_ok = ipopt_status_is_acceptable(ipopt_status) or (
        ipopt_status == "User_Requested_Stop" and result["stopped_early"]
    )
    acceptable_result = (
        ipopt_status_ok and np.isfinite(result["f"]) and final_eval["valid"]
    )
    if not acceptable_result:
        logger.warning("Final IPOPT point is not acceptable for saving as final.csv.")

    if not args.no_plot:
        vis_compare_seqs(
            [final_eval["t"], final_eval["t"], final_eval["t"]],
            [final_eval["q"], final_eval["dq"], final_eval["ddq"]],
            ["q", "qd", "qdd"],
            ["time"],
        )
        plt.show()

    if experiment_dir is not None and acceptable_result:
        save_traj_csv(
            experiment_dir,
            "final",
            final_eval["t"],
            final_eval["q"],
            final_eval["dq"],
            final_eval["ddq"],
        )


if __name__ == "__main__":
    main()
