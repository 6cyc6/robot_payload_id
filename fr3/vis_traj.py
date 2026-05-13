#!/usr/bin/env python3
"""Visualize FR3 excitation trajectories saved under fr3/logs.

The script loads trajectory CSV files with columns:
    t, q_0..q_N, dq_0..dq_N, ddq_0..ddq_N

It draws one figure each for joint positions, velocities, and accelerations, with
per-joint boundary lines when limits are available.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_ROOT = SCRIPT_DIR / "logs"
DEFAULT_URDF = SCRIPT_DIR / "robot_description/fr3_description/fr3.urdf"


@dataclass
class TrajectoryData:
    name: str
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


@dataclass
class JointLimits:
    position_lower: np.ndarray | None = None
    position_upper: np.ndarray | None = None
    velocity_abs: np.ndarray | None = None
    acceleration_abs: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FR3 excitation trajectory position, velocity, and acceleration."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=(
            "Trajectory CSV file, trajectory directory, or log root. Defaults to "
            "the local fr3/logs directory and uses the latest run directory "
            "containing CSV files."
        ),
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        default=None,
        help=(
            "CSV stem/name to plot, e.g. best_condition. Can be passed multiple times. "
            "By default all CSV trajectories in the selected run directory are plotted."
        ),
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=None,
        help=(
            "URDF to read joint position/velocity limits from. If omitted, the script "
            "uses robot_urdf_path from args.yaml when available, then falls back to "
            "the local FR3 URDF."
        ),
    )
    parser.add_argument(
        "--limit-scale",
        type=float,
        default=1.0,
        help="Scale position, velocity, and acceleration bounds before plotting.",
    )
    parser.add_argument(
        "--use-soft-position-limits",
        action="store_true",
        help="Use URDF safety_controller soft limits when present.",
    )
    parser.add_argument(
        "--acceleration-limits",
        type=str,
        default=None,
        help=(
            "Acceleration limits in rad/s^2. Use one scalar for all joints or a "
            "comma-separated list, e.g. '15,7.5,10,12.5,15,20,20'. If omitted, "
            "acceleration bounds are only drawn when the URDF contains acceleration "
            "limit attributes."
        ),
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of samples when plotting Fourier .npy trajectory folders.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save positions.png, velocities.png, accelerations.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows. Useful together with --save-dir.",
    )
    return parser.parse_args()


def numeric_suffix(name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", name)
    if match is None:
        raise ValueError(f"Column {name!r} does not match {prefix}_<index>.")
    return int(match.group(1))


def columns_for(data: np.ndarray, prefix: str) -> list[str]:
    names = list(data.dtype.names or [])
    columns = [
        name for name in names if re.fullmatch(rf"{re.escape(prefix)}_\d+", name)
    ]
    return sorted(columns, key=lambda name: numeric_suffix(name, prefix))


def column_stack(data: np.ndarray, columns: Iterable[str]) -> np.ndarray:
    return np.column_stack([np.atleast_1d(data[column]) for column in columns])


def load_csv_trajectory(path: Path) -> TrajectoryData:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"{path} does not look like a named-column CSV file.")

    q_columns = columns_for(data, "q")
    dq_columns = columns_for(data, "dq")
    ddq_columns = columns_for(data, "ddq")
    if not q_columns or not dq_columns or not ddq_columns:
        raise ValueError(
            f"{path} must contain q_*, dq_*, and ddq_* columns. "
            f"Found columns: {data.dtype.names}"
        )

    t = np.atleast_1d(data["t"])
    q = column_stack(data, q_columns)
    dq = column_stack(data, dq_columns)
    ddq = column_stack(data, ddq_columns)
    return TrajectoryData(name=path.stem, t=t, q=q, dq=dq, ddq=ddq)


def load_fourier_trajectory(path: Path, num_timesteps: int) -> TrajectoryData:
    a = np.load(path / "a_value.npy")
    b = np.load(path / "b_value.npy")
    q0 = np.load(path / "q0_value.npy")

    if (path / "omega.npy").exists():
        omega = float(np.load(path / "omega.npy").reshape(-1)[0])
        duration = 2.0 * np.pi / omega
    else:
        duration = 10.0
        omega = 2.0 * np.pi / duration

    t = np.linspace(0.0, duration, num_timesteps, endpoint=True)
    harmonic_ids = np.arange(1, a.shape[1] + 1, dtype=float)
    omega_l = omega * harmonic_ids
    phase = t[:, None] * omega_l[None, :]
    sin_part = np.sin(phase)
    cos_part = np.cos(phase)

    q = sin_part @ a.T + cos_part @ b.T + q0
    dq = (omega_l[None, :] * cos_part) @ a.T - (omega_l[None, :] * sin_part) @ b.T
    ddq = (
        -((omega_l[None, :] ** 2) * sin_part) @ a.T
        - ((omega_l[None, :] ** 2) * cos_part) @ b.T
    )
    return TrajectoryData(name=path.name, t=t, q=q, dq=dq, ddq=ddq)


def run_dir_mtime(path: Path) -> float:
    files = list(path.glob("*.csv"))
    if not files:
        return path.stat().st_mtime
    return max(file.stat().st_mtime for file in files)


def latest_run_dir(log_root: Path) -> Path:
    candidates = [
        path
        for path in log_root.rglob("*")
        if path.is_dir() and list(path.glob("*.csv"))
    ]
    if not candidates and list(log_root.glob("*.csv")):
        return log_root
    if not candidates:
        raise FileNotFoundError(f"No trajectory CSV files found under {log_root}.")
    return max(candidates, key=run_dir_mtime)


def select_csv_paths(
    path: Path, trajectory_names: list[str] | None
) -> tuple[Path, list[Path]]:
    if path.is_file():
        return path.parent, [path]

    if (path / "a_value.npy").exists():
        return path, []

    run_dir = path if list(path.glob("*.csv")) else latest_run_dir(path)
    csv_paths = sorted(run_dir.glob("*.csv"))
    if trajectory_names:
        wanted = {Path(name).stem for name in trajectory_names}
        csv_paths = [csv_path for csv_path in csv_paths if csv_path.stem in wanted]
        missing = wanted.difference({csv_path.stem for csv_path in csv_paths})
        if missing:
            raise FileNotFoundError(
                f"Could not find requested trajectory CSV(s) in {run_dir}: "
                + ", ".join(sorted(missing))
            )
    return run_dir, csv_paths


def resolve_urdf_path(run_dir: Path, explicit_urdf_path: Path | None) -> Path | None:
    if explicit_urdf_path is not None:
        return explicit_urdf_path.expanduser().resolve()

    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        with args_yaml.open("r", encoding="utf-8") as f:
            args_data = yaml.safe_load(f) or {}
        urdf_path = args_data.get("robot_urdf_path")
        if urdf_path:
            path = Path(urdf_path).expanduser()
            if not path.is_absolute():
                candidates = [
                    (run_dir / path).resolve(),
                    (Path.cwd() / path).resolve(),
                    (SCRIPT_DIR / path).resolve(),
                    (REPO_ROOT / path).resolve(),
                ]
                path = next(
                    (candidate for candidate in candidates if candidate.exists()),
                    candidates[0],
                )
            return path

    fallback = DEFAULT_URDF
    return fallback.resolve() if fallback.exists() else None


def parse_tag_attrs(tag_text: str) -> dict[str, str]:
    attr_pattern = re.compile(r'([A-Za-z_][\w:.-]*)\s*=\s*"([^"]*)"')
    return {match.group(1): match.group(2) for match in attr_pattern.finditer(tag_text)}


def iter_urdf_joint_records(
    urdf_path: Path,
) -> Iterable[tuple[str, dict[str, str], dict[str, str]]]:
    """Yield joint type, limit attributes, and safety attributes.

    Some local FR3 URDF variants contain a small XML mismatch after the arm joints.
    Strict XML parsing is preferred, but the regex fallback still recovers the joint
    limit tags before/around that malformed section.
    """
    try:
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            joint_type = joint.attrib.get("type", "")
            limit = joint.find("limit")
            if limit is None:
                continue
            safety = joint.find("safety_controller")
            yield joint_type, dict(limit.attrib), dict(
                safety.attrib
            ) if safety is not None else {}
        return
    except ET.ParseError:
        pass

    text = urdf_path.read_text(encoding="utf-8")
    joint_pattern = re.compile(r"<joint\b(?P<attrs>[^>]*)>(?P<body>.*?)</joint>", re.S)
    limit_pattern = re.compile(r"<limit\b(?P<attrs>[^>]*)/?>", re.S)
    safety_pattern = re.compile(r"<safety_controller\b(?P<attrs>[^>]*)/?>", re.S)

    for joint_match in joint_pattern.finditer(text):
        joint_attrs = parse_tag_attrs(joint_match.group("attrs"))
        limit_match = limit_pattern.search(joint_match.group("body"))
        if limit_match is None:
            continue
        safety_match = safety_pattern.search(joint_match.group("body"))
        yield (
            joint_attrs.get("type", ""),
            parse_tag_attrs(limit_match.group("attrs")),
            parse_tag_attrs(safety_match.group("attrs")) if safety_match else {},
        )


def parse_urdf_limits(
    urdf_path: Path | None,
    num_joints: int,
    limit_scale: float,
    use_soft_position_limits: bool,
) -> JointLimits:
    if urdf_path is None or not urdf_path.exists():
        return JointLimits()

    lower_values: list[float] = []
    upper_values: list[float] = []
    velocity_values: list[float] = []
    acceleration_values: list[float] = []

    for joint_type, limit_attrs, safety_attrs in iter_urdf_joint_records(urdf_path):
        if joint_type not in {"revolute", "continuous"}:
            continue

        lower = limit_attrs.get("lower")
        upper = limit_attrs.get("upper")
        velocity = limit_attrs.get("velocity")
        acceleration = (
            limit_attrs.get("acceleration")
            or limit_attrs.get("drake:acceleration")
            or next(
                (
                    value
                    for key, value in limit_attrs.items()
                    if key.endswith(":acceleration") or key.endswith("}acceleration")
                ),
                None,
            )
        )

        if use_soft_position_limits:
            lower = safety_attrs.get("soft_lower_limit", lower)
            upper = safety_attrs.get("soft_upper_limit", upper)

        if lower is None or upper is None or velocity is None:
            continue

        lower_values.append(float(lower))
        upper_values.append(float(upper))
        velocity_values.append(abs(float(velocity)))
        if acceleration is not None:
            acceleration_values.append(abs(float(acceleration)))

        if len(lower_values) == num_joints:
            break

    if len(lower_values) < num_joints:
        return JointLimits()

    limits = JointLimits(
        position_lower=limit_scale * np.asarray(lower_values[:num_joints], dtype=float),
        position_upper=limit_scale * np.asarray(upper_values[:num_joints], dtype=float),
        velocity_abs=limit_scale
        * np.asarray(velocity_values[:num_joints], dtype=float),
    )
    if len(acceleration_values) >= num_joints:
        limits.acceleration_abs = limit_scale * np.asarray(
            acceleration_values[:num_joints], dtype=float
        )
    return limits


def parse_acceleration_limits(
    value: str | None, num_joints: int, limit_scale: float
) -> np.ndarray | None:
    if value is None:
        return None

    values = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    if len(values) == 1:
        values = np.repeat(values[0], num_joints)
    if len(values) != num_joints:
        raise ValueError(
            f"--acceleration-limits must provide 1 value or {num_joints} values; "
            f"got {len(values)}."
        )
    return limit_scale * np.abs(values)


def boundary_label(signal_name: str, side: str) -> str:
    return f"{signal_name} {side} bound"


def plot_joint_series(
    trajectories: list[TrajectoryData],
    series_name: str,
    values_attr: str,
    ylabel: str,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> plt.Figure:
    num_joints = getattr(trajectories[0], values_attr).shape[1]
    fig_height = max(7.0, 1.8 * num_joints)
    fig, axes = plt.subplots(num_joints, 1, figsize=(12.0, fig_height), sharex=True)
    axes = np.atleast_1d(axes)

    for joint_idx, ax in enumerate(axes):
        for trajectory in trajectories:
            values = getattr(trajectory, values_attr)
            ax.plot(
                trajectory.t, values[:, joint_idx], linewidth=1.4, label=trajectory.name
            )

        if lower is not None and upper is not None:
            ax.axhline(
                lower[joint_idx],
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=boundary_label(series_name, "lower") if joint_idx == 0 else None,
            )
            ax.axhline(
                upper[joint_idx],
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label=boundary_label(series_name, "upper") if joint_idx == 0 else None,
            )

        ax.set_ylabel(f"J{joint_idx + 1}\n{ylabel}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Joint {series_name}", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.97))
    return fig


def load_trajectories(args: argparse.Namespace) -> tuple[Path, list[TrajectoryData]]:
    path = args.path.expanduser()
    run_dir, csv_paths = select_csv_paths(path, args.trajectory)

    if csv_paths:
        trajectories = [load_csv_trajectory(csv_path) for csv_path in csv_paths]
    elif (run_dir / "a_value.npy").exists():
        trajectories = [load_fourier_trajectory(run_dir, args.num_timesteps)]
    else:
        raise FileNotFoundError(f"No supported trajectory files found at {path}.")

    num_joints = trajectories[0].q.shape[1]
    for trajectory in trajectories:
        if trajectory.q.shape[1] != num_joints:
            raise ValueError("All plotted trajectories must have the same joint count.")
    return run_dir, trajectories


def main() -> None:
    args = parse_args()
    run_dir, trajectories = load_trajectories(args)
    num_joints = trajectories[0].q.shape[1]

    urdf_path = resolve_urdf_path(run_dir, args.urdf_path)
    limits = parse_urdf_limits(
        urdf_path=urdf_path,
        num_joints=num_joints,
        limit_scale=args.limit_scale,
        use_soft_position_limits=args.use_soft_position_limits,
    )
    explicit_acceleration_limits = parse_acceleration_limits(
        args.acceleration_limits, num_joints, args.limit_scale
    )
    if explicit_acceleration_limits is not None:
        limits.acceleration_abs = explicit_acceleration_limits

    print(f"Loaded {len(trajectories)} trajectory file(s) from {run_dir}")
    for trajectory in trajectories:
        print(
            f"  {trajectory.name}: {len(trajectory.t)} samples, "
            f"{trajectory.q.shape[1]} joints, duration {trajectory.t[-1] - trajectory.t[0]:.3f}s"
        )
    if urdf_path is not None:
        print(f"Using limits from {urdf_path}")
    if limits.acceleration_abs is None:
        print(
            "No acceleration limits were found; pass --acceleration-limits to draw "
            "acceleration boundary lines."
        )

    figures = {
        "positions": plot_joint_series(
            trajectories=trajectories,
            series_name="positions",
            values_attr="q",
            ylabel="rad",
            lower=limits.position_lower,
            upper=limits.position_upper,
        ),
        "velocities": plot_joint_series(
            trajectories=trajectories,
            series_name="velocities",
            values_attr="dq",
            ylabel="rad/s",
            lower=-limits.velocity_abs if limits.velocity_abs is not None else None,
            upper=limits.velocity_abs,
        ),
        "accelerations": plot_joint_series(
            trajectories=trajectories,
            series_name="accelerations",
            values_attr="ddq",
            ylabel="rad/s^2",
            lower=-limits.acceleration_abs
            if limits.acceleration_abs is not None
            else None,
            upper=limits.acceleration_abs,
        ),
    }

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            out_path = args.save_dir / f"{name}.png"
            fig.savefig(out_path, dpi=180)
            print(f"Saved {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
