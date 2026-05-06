import argparse
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

for thread_env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import numpy as np
import wandb
import yaml

from pydrake.all import AugmentedLagrangianNonsmooth

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.environment import create_arm
from robot_payload_id.optimization import (
    CostFunction,
    ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric,
)
from robot_payload_id.optimization import (
    optimal_experiment_design_fourier as oed_fourier,
)
from robot_payload_id.symbolic import eval_expression_mat
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes, name_constraint


DEFAULT_SYSTEM_IDENTIFICATION_ROOT = Path(
    "/home/ikun/github_repo/6cyc6/system-identification"
)
DEFAULT_FR3_URDF = (
    DEFAULT_SYSTEM_IDENTIFICATION_ROOT
    / "robot_description"
    / "fr3_description"
    / "fr3.urdf"
)
DEFAULT_FR3_Q0 = np.array(
    [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]
)


class FR3ExcitationOptimizer(ExcitationTrajectoryOptimizerFourierBlackBoxALNumeric):
    """Repo-native Fourier AL optimizer with FR3 camera and workspace constraints."""

    def __init__(
        self,
        *args,
        collision_checker: Any,
        link_y_lower: float,
        link_y_upper: float,
        link_z_lower: float,
        collision_constraint_stride: int,
        **kwargs,
    ):
        self._fr3_collision_checker = collision_checker
        self._link_y_lower = float(link_y_lower)
        self._link_y_upper = float(link_y_upper)
        self._link_z_lower = float(link_z_lower)
        self._collision_constraint_stride = max(1, int(collision_constraint_stride))

        original_create_arm = oed_fourier.create_arm

        def create_arm_without_meshcat(
            arm_file_path: str,
            num_joints: int,
            time_step: float = 0.0,
            use_meshcat: bool = False,
        ):
            return original_create_arm(
                arm_file_path=arm_file_path,
                num_joints=num_joints,
                time_step=time_step,
                use_meshcat=False,
            )

        oed_fourier.create_arm = create_arm_without_meshcat
        try:
            super().__init__(*args, **kwargs)
        finally:
            oed_fourier.create_arm = original_create_arm

    def _simulate_traj_and_log_recording(
        self, name: str, var_values: np.ndarray
    ) -> None:
        logging.debug("Skipping Meshcat checkpoint recording for %s.", name)

    @property
    def collision_checker(self) -> Any:
        return self._fr3_collision_checker

    def _add_collision_constraints(self, min_distance: float = 0.01) -> None:
        """Add only the FR3 camera-box and workspace wall constraints."""
        joint_positions_sym = self._compute_joint_positions(self._symbolic_vars)
        time_indices = np.arange(
            0,
            self._num_timesteps,
            self._collision_constraint_stride,
            dtype=int,
        )
        joint_positions_sampled_sym = joint_positions_sym[time_indices]

        def fr3_path_constraints(var_values: np.ndarray) -> np.ndarray:
            q = eval_expression_mat(
                joint_positions_sampled_sym,
                self._symbolic_vars,
                var_values,
            )
            camera_values = (
                self._fr3_collision_checker.minimum_distance_constraint_values(q)
            )
            wall_values = self._fr3_collision_checker.robot_link_wall_margins(
                q,
                y_lower=self._link_y_lower,
                y_upper=self._link_y_upper,
                z_lower=self._link_z_lower,
            )
            return np.concatenate([camera_values, wall_values.reshape(-1)])

        num_samples = len(time_indices)
        lower_bounds = np.zeros(4 * num_samples)
        upper_bounds = np.concatenate(
            [
                np.ones(num_samples),
                np.full(3 * num_samples, np.inf),
            ]
        )
        name_constraint(
            self._prog.AddConstraint(
                func=fr3_path_constraints,
                lb=lower_bounds,
                ub=upper_bounds,
                vars=self._symbolic_vars,
            ),
            (
                "fr3CameraAndWorkspaceBounds_"
                f"stride_{self._collision_constraint_stride}"
            ),
        )


@dataclass
class FR3OptimizerFactory:
    args: argparse.Namespace
    model_path: Path
    logging_path: Optional[Path]

    def construct_optimizer(self) -> FR3ExcitationOptimizer:
        np.random.seed(self.args.seed)
        if wandb.run is None:
            wandb.init(project="robot_payload_id", mode="disabled")
            os.makedirs(wandb.run.dir, exist_ok=True)
        bootstrap_system_identification(self.args.system_identification_root)
        from system_identification.drake.camera_collision import (
            DrakeCameraCollisionChecker,
        )

        collision_checker = make_collision_checker(
            self.args, DrakeCameraCollisionChecker
        )
        arm_components = create_arm(arm_file_path=str(self.model_path), num_joints=7)
        plant = arm_components.plant
        plant_context = plant.GetMyContextFromRoot(
            arm_components.diagram.CreateDefaultContext()
        )
        robot_model_instance_idx = plant.GetModelInstanceByName("arm")

        return FR3ExcitationOptimizer(
            num_joints=7,
            cost_function=self.args.cost_function,
            num_fourier_terms=self.args.num_fourier_terms,
            omega=self.args.omega,
            num_timesteps=self.args.num_timesteps,
            time_horizon=self.args.max_time_horizon,
            plant=plant,
            plant_context=plant_context,
            robot_model_instance_idx=robot_model_instance_idx,
            max_al_iterations=self.args.max_al_iterations,
            budget_per_iteration=self.args.budget,
            mu_initial=self.args.mu_initial,
            mu_multiplier=self.args.mu_multiplier,
            mu_max=self.args.mu_max,
            model_path=str(self.model_path),
            add_rotor_inertia=self.args.add_rotor_inertia,
            add_reflected_inertia=self.args.add_reflected_inertia,
            add_viscous_friction=self.args.add_viscous_friction,
            add_dynamic_dry_friction=self.args.add_dynamic_dry_friction,
            payload_only=self.args.payload_only,
            include_endpoint_constraints=not self.args.not_add_endpoint_constraints,
            nevergrad_method=self.args.nevergrad_method,
            traj_initial=self.args.traj_initial,
            initial_guess_scaling=self.args.initial_guess_scaling,
            logging_path=self.logging_path,
            collision_checker=collision_checker,
            link_y_lower=self.args.link_y_lower,
            link_y_upper=self.args.link_y_upper,
            link_z_lower=self.args.link_z_lower,
            collision_constraint_stride=self.args.collision_constraint_stride,
        )


@dataclass
class FR3AugmentedLagrangianFactory:
    optimizer_factory: FR3OptimizerFactory

    def __call__(self) -> AugmentedLagrangianNonsmooth:
        optimizer = self.optimizer_factory.construct_optimizer()
        return AugmentedLagrangianNonsmooth(
            prog=optimizer._prog,
            include_x_bounds=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an FR3 excitation trajectory with this repo's native "
            "Fourier black-box augmented-Lagrangian solver, using only the FR3 "
            "camera-box collision constraints and workspace bounds."
        )
    )
    parser.add_argument(
        "--use_one_link_arm",
        action="store_true",
        help=(
            "Accepted for CLI compatibility with design_optimal_excitation_trajectories.py; "
            "ignored because this script always optimizes the 7-DOF FR3."
        ),
    )
    parser.add_argument(
        "--system_identification_root",
        type=Path,
        default=DEFAULT_SYSTEM_IDENTIFICATION_ROOT,
        help="Path to the referenced system-identification repo.",
    )
    parser.add_argument(
        "--robot_urdf_path",
        type=Path,
        default=DEFAULT_FR3_URDF,
        help="Original FR3 URDF used to build the collision checker.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="black_box",
        choices=["black_box"],
        help="Optimizer to use. This FR3 script supports the repo's black_box AL path.",
    )
    parser.add_argument(
        "--cost_function",
        type=CostFunction,
        default=CostFunction.CONDITION_NUMBER_AND_E_OPTIMALITY,
        choices=list(CostFunction),
        help="Cost function to use.",
    )
    parser.add_argument(
        "--num_fourier_terms",
        type=int,
        default=5,
        help="Number of Fourier terms to use.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.3 * np.pi,
        help="Frequency of the Fourier series trajectory.",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="The number of timesteps to use.",
    )
    parser.add_argument(
        "--min_time_horizon",
        type=float,
        default=10,
        help="Kept for argument compatibility with the repo script.",
    )
    parser.add_argument(
        "--max_time_horizon",
        type=float,
        default=10,
        help="The time horizon/ duration of the trajectory.",
    )
    parser.add_argument(
        "--max_al_iterations",
        type=int,
        default=10,
        help="Maximum number of augmented Lagrangian iterations.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3000,
        help="Budget per augmented Lagrangian iteration.",
    )
    parser.add_argument(
        "--mu_initial",
        type=float,
        default=5.0,
        help="Initial value of the augmented Lagrangian parameter.",
    )
    parser.add_argument(
        "--mu_multiplier",
        type=float,
        default=1.5,
        help="Multiplier for the augmented Lagrangian parameter.",
    )
    parser.add_argument(
        "--mu_max",
        type=float,
        default=1e3,
        help="Maximum value of the augmented Lagrangian parameter.",
    )
    parser.add_argument(
        "--nevergrad_method",
        type=str,
        default="NGOpt",
        help="Nevergrad method to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Kept for argument compatibility. This FR3 constraint wrapper uses 1.",
    )
    parser.add_argument(
        "--logging_path",
        type=Path,
        default=Path("logs/traj"),
        help="Path to the directory to save the logs to.",
    )
    parser.add_argument(
        "--traj_initial",
        type=Path,
        default=None,
        help="Path to an initial Fourier trajectory.",
    )
    parser.add_argument(
        "--add_rotor_inertia",
        action="store_true",
        help="Add reflected rotor inertia to the optimization.",
    )
    parser.add_argument(
        "--add_reflected_inertia",
        action="store_true",
        help="Add reflected inertia to the optimization.",
    )
    parser.add_argument(
        "--add_viscous_friction",
        action="store_true",
        help="Add viscous friction to the optimization.",
    )
    parser.add_argument(
        "--add_dynamic_dry_friction",
        action="store_true",
        help="Add dynamic dry friction to the optimization.",
    )
    parser.add_argument(
        "--not_add_endpoint_constraints",
        action="store_true",
        help="Disable zero velocity/acceleration endpoint constraints.",
    )
    parser.add_argument(
        "--payload_only",
        action="store_true",
        help="Only consider the 10 inertial parameters of the last link.",
    )
    parser.add_argument(
        "--initial_guess_scaling",
        type=float,
        default=1.0,
        help="Scaling for the random initial guess.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["disabled", "online", "offline"],
        help="WandB mode.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    parser.add_argument(
        "--collision_constraint_stride",
        type=int,
        default=1,
        help="Apply camera and workspace constraints every N trajectory samples.",
    )
    parser.add_argument("--drake_min_distance", type=float, default=0.02)
    parser.add_argument("--drake_robot_sphere_radius", type=float, default=0.05)
    parser.add_argument("--drake_robot_link_samples", type=int, default=5)
    parser.add_argument("--drake_camera_chamfer_radius", type=float, default=0.02)
    parser.add_argument(
        "--drake_xy_prism_height",
        type=float,
        default=None,
        help="Optional full z height for XY-prism camera obstacles.",
    )
    parser.add_argument(
        "--drake_physical_camera_height",
        action="store_true",
        help="Use the physical camera-box z height even if xy_prism_height is set.",
    )
    parser.add_argument("--link_y_lower", type=float, default=-0.5)
    parser.add_argument("--link_y_upper", type=float, default=0.4)
    parser.add_argument("--link_z_lower", type=float, default=0.0)
    return parser.parse_args()


def bootstrap_system_identification(system_identification_root: Path) -> Path:
    root = system_identification_root.expanduser().resolve()
    if not (root / "system_identification").is_dir():
        raise FileNotFoundError(f"Missing system_identification package under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _rewrite_model_name(name: str) -> str:
    return name.replace("panda_link", "link").replace("panda_joint", "joint")


def write_repo_compatible_fr3_urdf(
    source_urdf: Path,
    output_urdf: Path,
) -> None:
    source_urdf = source_urdf.expanduser().resolve()
    output_urdf.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    root.set("name", "arm")

    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)

    for joint in root.findall("joint"):
        for safety_controller in list(joint.findall("safety_controller")):
            joint.remove(safety_controller)

    for transmission in list(root.findall("transmission")):
        root.remove(transmission)

    for elem in root.iter():
        for attr in ("name", "link"):
            if attr in elem.attrib:
                elem.set(attr, _rewrite_model_name(elem.attrib[attr]))
        if elem.tag == "mesh" and "filename" in elem.attrib:
            filename = elem.attrib["filename"]
            if not filename.startswith(("package://", "file://", "/")):
                elem.set("filename", str((source_urdf.parent / filename).resolve()))

    for i in range(7):
        transmission = ET.SubElement(root, "transmission", {"name": f"tran{i + 1}"})
        ET.SubElement(
            transmission, "type"
        ).text = "transmission_interface/SimpleTransmission"
        joint = ET.SubElement(transmission, "joint", {"name": f"joint{i + 1}"})
        ET.SubElement(joint, "hardwareInterface").text = "EffortJointInterface"
        actuator = ET.SubElement(transmission, "actuator", {"name": f"joint{i + 1}"})
        ET.SubElement(actuator, "hardwareInterface").text = "EffortJointInterface"
        ET.SubElement(actuator, "mechanicalReduction").text = "1"

    tree.write(output_urdf, encoding="unicode", xml_declaration=True)


def write_fr3_model_directives(
    model_path: Path,
    normalized_urdf_path: Path,
) -> None:
    default_joint_positions = {
        f"joint{i + 1}": [float(DEFAULT_FR3_Q0[i])] for i in range(7)
    }
    directives = {
        "directives": [
            {
                "add_model": {
                    "name": "arm",
                    "file": normalized_urdf_path.resolve().as_uri(),
                    "default_joint_positions": default_joint_positions,
                }
            },
            {
                "add_weld": {
                    "parent": "world",
                    "child": "arm::link0",
                }
            },
        ]
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(directives, f, sort_keys=False)


def yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CostFunction):
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


def make_collision_checker(args: argparse.Namespace, checker_cls: Any) -> Any:
    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    return checker_cls(
        robot_name="fr3",
        robot_urdf_path=args.robot_urdf_path,
        min_distance=args.drake_min_distance,
        robot_sphere_radius=args.drake_robot_sphere_radius,
        robot_link_samples=args.drake_robot_link_samples,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
    )


def save_traj_csv(path: Path, joint_data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        [
            joint_data.sample_times_s,
            joint_data.joint_positions,
            joint_data.joint_velocities,
            joint_data.joint_accelerations,
        ]
    )
    njoints = joint_data.joint_positions.shape[1]
    header = (
        ["t"]
        + [f"q_{i}" for i in range(njoints)]
        + [f"dq_{i}" for i in range(njoints)]
        + [f"ddq_{i}" for i in range(njoints)]
    )
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def evaluate_constraints(
    joint_positions: np.ndarray,
    collision_checker: Any,
    args: argparse.Namespace,
) -> dict[str, float]:
    stride = max(1, int(args.collision_constraint_stride))
    q_sampled = joint_positions[::stride]
    collision_values = collision_checker.minimum_distance_constraint_values(q_sampled)
    wall_values = collision_checker.robot_link_wall_margins(
        q_sampled,
        y_lower=args.link_y_lower,
        y_upper=args.link_y_upper,
        z_lower=args.link_z_lower,
    )

    return {
        "constraint_stride": stride,
        "num_constraint_samples": int(len(q_sampled)),
        "min_camera_constraint_value": float(np.min(collision_values)),
        "min_camera_clearance": float(collision_checker.min_clearance(q_sampled)),
        "min_y_lower_margin": float(np.min(wall_values[:, 0])),
        "min_y_upper_margin": float(np.min(wall_values[:, 1])),
        "min_z_margin": float(np.min(wall_values[:, 2])),
    }


def resolve_logging_path(logging_path: Optional[Path]) -> Optional[Path]:
    if logging_path is None:
        return None
    path = logging_path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def optimize_with_workers(
    optimizer: FR3ExcitationOptimizer,
    al_factory: FR3AugmentedLagrangianFactory,
    num_workers: int,
    mu_initial: float,
) -> FourierSeriesTrajectoryAttributes:
    if num_workers == 1:
        return optimizer.optimize()

    logging.info("Starting FR3 parallel optimization with %d workers.", num_workers)
    logging.info(
        "The first progress-bar tick appears after one full worker batch returns; "
        "with %d workers, that is %d candidate evaluations.",
        num_workers,
        num_workers,
    )
    num_lambda = optimizer._ng_al.compute_num_lambda(optimizer._prog)
    lambda_initial = np.zeros(num_lambda)
    x_val, _, _ = optimizer._ng_al.solve(
        prog_or_al_factory=al_factory,
        x_init=optimizer._initial_guess,
        lambda_val=lambda_initial,
        mu=mu_initial,
        nevergrad_set_bounds=True,
        num_workers=num_workers,
        log_check_point_callback=optimizer._extract_and_log_optimization_result,
    )
    return optimizer._extract_fourier_trajectory_attributes(x_val)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    if args.num_workers > 1:
        mp.set_start_method("spawn", force=True)

    if args.use_one_link_arm:
        logging.warning(
            "--use_one_link_arm is ignored by gen_fr3_excitation_traj.py; "
            "optimizing the 7-DOF FR3."
        )

    args.system_identification_root = bootstrap_system_identification(
        args.system_identification_root
    )

    args.robot_urdf_path = args.robot_urdf_path.expanduser().resolve()
    logging_path = resolve_logging_path(args.logging_path)
    temporary_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if logging_path is None:
        temporary_dir = tempfile.TemporaryDirectory(prefix="fr3_excitation_model_")
        generated_model_dir = Path(temporary_dir.name)
    else:
        generated_model_dir = logging_path / "generated_model"

    normalized_urdf_path = generated_model_dir / "fr3_repo_names.urdf"
    model_path = generated_model_dir / "fr3_repo_names.dmd.yaml"
    write_repo_compatible_fr3_urdf(args.robot_urdf_path, normalized_urdf_path)
    write_fr3_model_directives(model_path, normalized_urdf_path)

    np.random.seed(args.seed)
    logging.info(
        "Using system-identification repo at %s", args.system_identification_root
    )
    logging.info("Using generated FR3 model directives at %s", model_path)
    logging.info(
        "FR3 workspace constraints: %s < y < %s, z > %s",
        args.link_y_lower,
        args.link_y_upper,
        args.link_z_lower,
    )

    run_name = f"fr3_excitation_design {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    wandb.init(
        project="robot_payload_id",
        name=run_name,
        config={**yaml_safe(vars(args)), "model_path": str(model_path)},
        mode=args.wandb_mode,
    )
    if wandb.run is not None:
        os.makedirs(wandb.run.dir, exist_ok=True)

    try:
        if logging_path is not None:
            logging_path.mkdir(parents=True, exist_ok=True)
            args_dict = {
                **yaml_safe(vars(args)),
                "model_path": str(model_path),
                "normalized_urdf_path": str(normalized_urdf_path),
            }
            if wandb.run is not None:
                args_dict["wandb_url"] = wandb.run.url
            with open(logging_path / "args.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(args_dict, f, sort_keys=True)

        optimizer_factory = FR3OptimizerFactory(
            args=args,
            model_path=model_path,
            logging_path=logging_path,
        )
        worker_optimizer_factory = FR3OptimizerFactory(
            args=args,
            model_path=model_path,
            logging_path=None,
        )
        optimizer = optimizer_factory.construct_optimizer()

        start = time.perf_counter()
        traj_attrs = optimize_with_workers(
            optimizer=optimizer,
            al_factory=FR3AugmentedLagrangianFactory(worker_optimizer_factory),
            num_workers=args.num_workers,
            mu_initial=args.mu_initial,
        )
        elapsed = time.perf_counter() - start
        logging.info("FR3 excitation optimization time cost: %.3f s", elapsed)

        if logging_path is not None:
            traj_attrs.log(logging_path=logging_path)
            final_path = logging_path / "final"
            final_path.mkdir(exist_ok=True)
            traj_attrs.log(logging_path=final_path)

            joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
                num_timesteps=args.num_timesteps,
                time_horizon=args.max_time_horizon,
                traj_attrs=traj_attrs,
                use_progress_bar=False,
            )
            save_traj_csv(logging_path / "final.csv", joint_data)
            constraint_report = evaluate_constraints(
                joint_data.joint_positions,
                optimizer.collision_checker,
                args,
            )
            constraint_report["optimization_time_s"] = elapsed
            with open(
                logging_path / "constraint_report.yaml", "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(yaml_safe(constraint_report), f, sort_keys=True)
            logging.info("Saved final trajectory to %s", logging_path)
    finally:
        wandb.finish()
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
