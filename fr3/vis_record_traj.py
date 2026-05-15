#!/usr/bin/env python3
"""Visualize recorded FR3 joint trajectory arrays."""

from __future__ import annotations

import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_DIR = Path("/home/galois/Downloads/trajectory_2")


def load_array(
    data_dir: Path, name: str, *, required: bool = True
) -> np.ndarray | None:
    path = data_dir / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return np.load(path)


def normalize_time(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("Time array is empty.")
    return t - t[0]


def shift_time(t: np.ndarray, t0: float) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size == 0:
        raise ValueError("Time array is empty.")
    return t - float(t0)


def check_samples(name: str, values: np.ndarray, times: np.ndarray) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (T, njoints), got {values.shape}.")
    if values.shape[0] != len(times):
        raise ValueError(
            f"{name} has {values.shape[0]} samples, but time has {len(times)}."
        )


def finite_joint_mask(values: np.ndarray) -> np.ndarray:
    return np.any(np.isfinite(values), axis=0)


def plot_joint_series(
    ax,
    times: np.ndarray,
    values: np.ndarray,
    *,
    joints: list[int],
    ylabel: str,
    title: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    label_prefix: str = "joint",
) -> bool:
    finite_mask = finite_joint_mask(values)
    plotted = False
    for joint in joints:
        if joint < 0 or joint >= values.shape[1]:
            raise ValueError(f"Joint index {joint} is outside shape {values.shape}.")
        if not finite_mask[joint]:
            continue
        ax.plot(
            times,
            values[:, joint],
            linestyle=linestyle,
            alpha=alpha,
            label=f"{label_prefix} {joint}",
        )
        plotted = True

    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(ncol=2, fontsize="small")
    else:
        ax.text(
            0.5,
            0.5,
            "No finite samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    return plotted


def plot_position_comparison_per_joint(
    sample_times: np.ndarray,
    joint_positions: np.ndarray,
    target_times: np.ndarray | None,
    target_positions: np.ndarray | None,
    *,
    joints: list[int],
) -> list[tuple[str, plt.Figure]]:
    figures = []
    recorded_finite = finite_joint_mask(joint_positions)
    target_finite = (
        finite_joint_mask(target_positions)
        if target_positions is not None
        else np.zeros(joint_positions.shape[1], dtype=bool)
    )

    for joint in joints:
        if joint < 0 or joint >= joint_positions.shape[1]:
            raise ValueError(
                f"Joint index {joint} is outside shape {joint_positions.shape}."
            )

        fig, ax = plt.subplots(figsize=(12, 5))
        plotted = False
        if recorded_finite[joint]:
            ax.plot(
                sample_times,
                joint_positions[:, joint],
                label="recorded",
                linewidth=1.8,
            )
            plotted = True

        if target_times is not None and target_positions is not None:
            if target_finite[joint]:
                ax.plot(
                    target_times,
                    target_positions[:, joint],
                    "--",
                    label="target",
                    linewidth=1.4,
                    alpha=0.8,
                )
                plotted = True
            else:
                ax.text(
                    0.99,
                    0.03,
                    "target has no finite samples",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize="small",
                    alpha=0.7,
                )

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No finite samples",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_title(f"Joint {joint} Position")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position [rad]")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        figures.append((f"joint_{joint}_position_recorded_vs_target", fig))

    return figures


def save_or_show(
    figures: list[tuple[str, plt.Figure]], save_dir: Path | None, show: bool
):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = save_dir / f"{name}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            print(f"Saved {path}")
    if show:
        plt.show()
    else:
        for _name, fig in figures:
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recorded FR3 joint position, velocity, acceleration, and torque arrays."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing recorded .npy arrays. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--joints",
        type=int,
        nargs="*",
        default=None,
        help="Joint indices to plot. Defaults to all joints in the file.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG output.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save/check plots without opening an interactive window.",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also plot recorded velocity, acceleration, and torque.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    sample_times_raw = load_array(data_dir, "sample_times_s.npy")
    joint_positions = load_array(data_dir, "joint_positions.npy")
    joint_velocities = load_array(data_dir, "joint_velocities.npy")
    joint_accelerations = load_array(data_dir, "joint_accelerations.npy")
    joint_torques = load_array(data_dir, "joint_torques.npy")
    target_times_raw = load_array(data_dir, "target_sample_times_s.npy", required=False)
    target_positions = load_array(
        data_dir, "target_joint_positions.npy", required=False
    )

    common_t0 = float(np.asarray(sample_times_raw).reshape(-1)[0])
    if target_times_raw is not None:
        common_t0 = min(common_t0, float(np.asarray(target_times_raw).reshape(-1)[0]))
    sample_times = shift_time(sample_times_raw, common_t0)

    check_samples("joint_positions.npy", joint_positions, sample_times)
    if args.include_extra:
        for name, values in (
            ("joint_velocities.npy", joint_velocities),
            ("joint_accelerations.npy", joint_accelerations),
            ("joint_torques.npy", joint_torques),
        ):
            check_samples(name, values, sample_times)

    njoints = joint_positions.shape[1]
    joints = args.joints if args.joints is not None else list(range(njoints))
    print(f"Loaded {data_dir}")
    print(
        f"Samples: {len(sample_times)}, joints: {njoints}, duration: {sample_times[-1]:.3f}s"
    )
    print(f"Plotting joints: {joints}")

    figures: list[tuple[str, plt.Figure]] = []

    target_times = None
    if target_times_raw is not None and target_positions is not None:
        target_times = shift_time(target_times_raw, common_t0)
        check_samples("target_joint_positions.npy", target_positions, target_times)

    figures.extend(
        plot_position_comparison_per_joint(
            sample_times,
            joint_positions,
            target_times,
            target_positions,
            joints=joints,
        )
    )

    if args.include_extra:
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_joint_series(
            ax,
            sample_times,
            joint_velocities,
            joints=joints,
            ylabel="velocity [rad/s]",
            title="Recorded Joint Velocities",
        )
        figures.append(("joint_velocities", fig))

        fig, ax = plt.subplots(figsize=(12, 7))
        plot_joint_series(
            ax,
            sample_times,
            joint_accelerations,
            joints=joints,
            ylabel="acceleration [rad/s^2]",
            title="Recorded Joint Accelerations",
        )
        figures.append(("joint_accelerations", fig))

        fig, ax = plt.subplots(figsize=(12, 7))
        plot_joint_series(
            ax,
            sample_times,
            joint_torques,
            joints=joints,
            ylabel="torque [Nm]",
            title="Recorded Joint Torques",
        )
        figures.append(("joint_torques", fig))

    save_or_show(figures, args.save_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
