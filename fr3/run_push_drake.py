#!/usr/bin/env python3
"""Replay a flip-cube joint trajectory with the FR3 pusher model in Drake."""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

import numpy as np

from loguru import logger
from pydrake.all import StartMeshcat

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_drake import (  # noqa: E402
    CAMERA_BOX_HEIGHT_SCALE,
    CAMERA_BOX_MARGIN_SCALE,
    build_visualization_diagram,
    hold_with_replay_button,
    play_trajectory,
    publish_robot_sample_markers,
    report_drake_collision,
    report_link_workspace_bounds,
    report_robot_self_collision,
    setup_robot_sample_markers,
    wait_for_meshcat_start,
)

DEFAULT_TRAJECTORY = SCRIPT_DIR / "traj_hit_q.csv"
DEFAULT_PUSHER_URDF = SCRIPT_DIR / "robot_description/fr3_description/fr3_pusher.urdf"
DEFAULT_BASE_Q = np.array(
    [
        0.64835417,
        0.30129686,
        0.12616242,
        -2.39593792,
        -0.08608791,
        2.69366050,
        1.63235939,
    ],
    dtype=float,
)
DEFAULT_S_CURVE_JOINT_INDEX = 0


def resolve_path(path: Path, *, default: Path) -> Path:
    path = default if path is None else Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        SCRIPT_DIR / path,
        SCRIPT_DIR.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {path}. Tried:\n  {tried}")


def sorted_columns(names: list[str], prefix: str) -> list[str]:
    columns = [name for name in names if name.startswith(prefix)]
    return sorted(columns, key=lambda name: int(name.rsplit("_", 1)[1]))


def load_push_joint_csv(
    path: Path,
    *,
    base_q: np.ndarray,
    s_curve_joint_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.dtype.names is None:
        raise ValueError(f"{path} does not look like a named-column CSV file.")

    names = list(raw.dtype.names)
    q_columns = sorted_columns(names, "joint_pos_") or sorted_columns(names, "q_")
    dq_columns = sorted_columns(names, "joint_vel_") or sorted_columns(names, "dq_")
    q_state_columns = sorted_columns(names, "joint_")

    if "time" in names:
        t = np.asarray(raw["time"], dtype=float)
    elif "time_s" in names:
        t = np.asarray(raw["time_s"], dtype=float)
    elif "t" in names:
        t = np.asarray(raw["t"], dtype=float)
    elif "step" in names and "sim_dt" in names:
        t = np.asarray(raw["step"], dtype=float) * np.asarray(
            raw["sim_dt"], dtype=float
        )
    else:
        t = np.arange(raw.shape[0], dtype=float)

    if q_columns or dq_columns:
        if not q_columns:
            raise ValueError(f"{path} must contain joint_pos_* or q_* columns.")
        if not dq_columns:
            raise ValueError(f"{path} must contain joint_vel_* or dq_* columns.")
        if len(q_columns) != len(dq_columns):
            raise ValueError(
                f"{path} has {len(q_columns)} position columns but "
                f"{len(dq_columns)} velocity columns."
            )

        q = np.column_stack([raw[name] for name in q_columns]).astype(float)
        dq = np.column_stack([raw[name] for name in dq_columns]).astype(float)
        if len(t) > 1:
            ddq = np.gradient(dq, t, axis=0, edge_order=2 if len(t) > 2 else 1)
        else:
            ddq = np.zeros_like(dq)
    elif q_state_columns:
        if len(q_state_columns) != 7:
            raise ValueError(
                f"{path} has {len(q_state_columns)} joint_* columns; expected 7."
            )

        q = np.column_stack([raw[name] for name in q_state_columns]).astype(float)
        if len(t) > 1:
            dq = np.gradient(q, t, axis=0, edge_order=2 if len(t) > 2 else 1)
            ddq = np.gradient(dq, t, axis=0, edge_order=2 if len(t) > 2 else 1)
        else:
            dq = np.zeros_like(q)
            ddq = np.zeros_like(q)
    else:
        delta_name = f"joint{s_curve_joint_index}_delta_rad"
        vel_name = f"joint{s_curve_joint_index}_vel_rad_s"
        acc_name = f"joint{s_curve_joint_index}_acc_rad_s2"
        if delta_name not in names or vel_name not in names:
            raise ValueError(
                f"{path} must contain either full joint columns or "
                f"{delta_name}/{vel_name} for S-curve playback."
            )

        base_q = np.asarray(base_q, dtype=float).reshape(-1)
        q = np.tile(base_q, (len(t), 1))
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
        q[:, s_curve_joint_index] += np.asarray(raw[delta_name], dtype=float)
        dq[:, s_curve_joint_index] = np.asarray(raw[vel_name], dtype=float)
        if acc_name in names:
            ddq[:, s_curve_joint_index] = np.asarray(raw[acc_name], dtype=float)
        elif len(t) > 1:
            ddq[:, s_curve_joint_index] = np.gradient(
                dq[:, s_curve_joint_index],
                t,
                edge_order=2 if len(t) > 2 else 1,
            )

    finite_mask = (
        np.isfinite(t)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(dq), axis=1)
    )
    if not np.all(finite_mask):
        logger.warning(
            "Dropping %d non-finite trajectory row(s).",
            int(len(finite_mask) - np.count_nonzero(finite_mask)),
        )
        t = t[finite_mask]
        q = q[finite_mask]
        dq = dq[finite_mask]
        ddq = ddq[finite_mask]

    if len(t) == 0:
        raise ValueError(f"{path} did not contain any finite trajectory samples.")
    t = t - t[0]
    return t, q, dq, ddq


def log_trajectory_stats(
    t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray
) -> None:
    logger.info(f"Samples: {len(t)}, joints: {q.shape[1]}, duration: {t[-1]:.3f}s")
    logger.info(f"Start q: {q[0]}")
    logger.info(f"End q: {q[-1]}")
    logger.info(f"Max abs dq per joint: {np.max(np.abs(dq), axis=0)}")
    logger.info(f"Max abs estimated ddq per joint: {np.max(np.abs(ddq), axis=0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay fr3/traj_hit_q.csv with the FR3 pusher URDF."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DEFAULT_TRAJECTORY,
        help="Push trajectory CSV. Defaults to fr3/traj_hit_q.csv.",
    )
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument(
        "--robot_urdf_path",
        type=Path,
        default=DEFAULT_PUSHER_URDF,
        help="Robot URDF path. Defaults to the local FR3 pusher URDF.",
    )
    parser.add_argument("--camera_collision_stride", type=int, default=5)
    parser.add_argument("--drake_min_distance", type=float, default=0.02)
    parser.add_argument("--drake_robot_sphere_radius", type=float, default=0.05)
    parser.add_argument("--drake_robot_link_samples", type=int, default=5)
    parser.add_argument("--drake_camera_chamfer_radius", type=float, default=0.02)
    parser.add_argument(
        "--camera_box_xy_scale",
        type=float,
        default=CAMERA_BOX_MARGIN_SCALE,
        help="Scale applied to camera obstacle x/y dimensions.",
    )
    parser.add_argument(
        "--camera_box_z_scale",
        type=float,
        default=CAMERA_BOX_HEIGHT_SCALE,
        help="Scale applied to camera obstacle z dimension.",
    )
    parser.add_argument("--link_x_lower", type=float, default=-0.15)
    parser.add_argument("--link_y_lower", type=float, default=-0.4)
    parser.add_argument("--link_y_upper", type=float, default=0.4)
    parser.add_argument("--link_z_lower", type=float, default=0.1)
    parser.add_argument("--disable_link_y_bounds", action="store_true")
    parser.add_argument(
        "--self_collision_check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check sampled FR3 self-collision sphere margins.",
    )
    parser.add_argument(
        "--self_collision_stride",
        type=int,
        default=None,
        help="Defaults to camera_collision_stride.",
    )
    parser.add_argument("--self_collision_clearance", type=float, default=0.0)
    parser.add_argument("--drake_xy_prism_height", type=float, default=None)
    parser.add_argument("--drake_physical_camera_height", action="store_true")
    parser.add_argument(
        "--base_q",
        type=float,
        nargs=7,
        default=DEFAULT_BASE_Q.tolist(),
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Nominal 7-DOF pose used when replaying joint*_delta_rad S-curve CSV files.",
    )
    parser.add_argument(
        "--s_curve_joint_index",
        type=int,
        default=DEFAULT_S_CURVE_JOINT_INDEX,
        help="Joint index driven by joint{index}_delta_rad in an S-curve CSV.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish the trajectory to Meshcat.",
    )
    parser.add_argument(
        "--start_button", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--start_button_name", type=str, default="Start push playback")
    parser.add_argument(
        "--replay_button", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--replay_button_name", type=str, default="Replay push trajectory"
    )
    parser.add_argument("--playback_stride", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--show_robot_samples", action="store_true")
    parser.add_argument("--robot_sample_marker_radius", type=float, default=None)
    parser.add_argument("--hold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stop_on_collision",
        action="store_true",
        help="Exit before playback if sampled camera/workspace/self collision fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_path = resolve_path(args.trajectory, default=DEFAULT_TRAJECTORY)
    robot_urdf_path = resolve_path(args.robot_urdf_path, default=DEFAULT_PUSHER_URDF)

    os.chdir(SCRIPT_DIR)
    t, q, dq, ddq = load_push_joint_csv(
        trajectory_path,
        base_q=np.asarray(args.base_q, dtype=float),
        s_curve_joint_index=args.s_curve_joint_index,
    )
    logger.info(f"Loaded push trajectory: {trajectory_path}")
    logger.info(f"Using robot URDF: {robot_urdf_path}")
    log_trajectory_stats(t, q, dq, ddq)

    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    checker, collision_values = report_drake_collision(
        q,
        t,
        robot_name=args.robot,
        robot_urdf_path=robot_urdf_path,
        stride=args.camera_collision_stride,
        min_distance=args.drake_min_distance,
        robot_sphere_radius=args.drake_robot_sphere_radius,
        robot_link_samples=args.drake_robot_link_samples,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
        camera_box_xy_scale=args.camera_box_xy_scale,
        camera_box_z_scale=args.camera_box_z_scale,
    )

    workspace_margins = np.array([np.inf])
    if not args.disable_link_y_bounds:
        workspace_margins = report_link_workspace_bounds(
            q,
            t,
            checker,
            stride=args.camera_collision_stride,
            x_lower=args.link_x_lower,
            y_lower=args.link_y_lower,
            y_upper=args.link_y_upper,
            z_lower=args.link_z_lower,
        )

    self_collision_margin = np.inf
    if args.self_collision_check:
        self_collision_margin = report_robot_self_collision(
            q,
            t,
            checker,
            stride=args.self_collision_stride or args.camera_collision_stride,
            clearance=args.self_collision_clearance,
        )

    if args.stop_on_collision and (
        np.min(collision_values) < 0.0
        or np.min(workspace_margins) < 0.0
        or self_collision_margin < args.self_collision_clearance
    ):
        raise SystemExit(1)

    if not args.visualize:
        return

    meshcat = StartMeshcat()
    meshcat.Delete()
    logger.info(f"Meshcat URL: {meshcat.web_url()}")

    (
        diagram,
        context,
        plant,
        plant_context,
        model_instance,
    ) = build_visualization_diagram(
        args.robot,
        meshcat,
        robot_urdf_path=robot_urdf_path,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
        camera_box_xy_scale=args.camera_box_xy_scale,
        camera_box_z_scale=args.camera_box_z_scale,
    )
    if q.shape[1] != plant.num_positions(model_instance):
        raise ValueError(
            f"Trajectory has {q.shape[1]} joints, but Drake model has "
            f"{plant.num_positions(model_instance)} positions."
        )

    context.SetTime(float(t[0]))
    plant.SetPositions(plant_context, model_instance, q[0])
    diagram.ForcedPublish(context)
    if args.show_robot_samples:
        setup_robot_sample_markers(
            meshcat,
            checker,
            marker_radius=args.robot_sample_marker_radius,
        )
        publish_robot_sample_markers(meshcat, checker, q[0])
    if args.start_button:
        wait_for_meshcat_start(meshcat, args.start_button_name)

    def play_once() -> None:
        play_trajectory(
            diagram,
            context,
            plant,
            plant_context,
            model_instance,
            t,
            q,
            playback_stride=args.playback_stride,
            speed=args.speed,
            meshcat=meshcat if args.show_robot_samples else None,
            robot_sample_checker=checker if args.show_robot_samples else None,
        )

    logger.info(
        f"Playing push trajectory in Meshcat at speed={args.speed}, "
        f"playback_stride={args.playback_stride}"
    )
    play_once()

    if args.hold and args.replay_button:
        hold_with_replay_button(meshcat, args.replay_button_name, play_once)
    elif args.hold:
        try:
            input("Press Enter to exit Meshcat playback...")
        except EOFError:
            logger.info("No stdin available; exiting Meshcat playback.")


if __name__ == "__main__":
    main()
