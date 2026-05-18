#!/usr/bin/env python3
"""Solve inertial-parameter SDP system ID for recorded FR3 joint data.

This is a local FR3 runner that intentionally leaves the repository's generic
`scripts/solve_inertial_param_sdp.py` untouched.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from pathlib import Path
from typing import Iterable

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    AutoDiffXd,
    DiagramBuilder,
    MultibodyForces_,
    MultibodyPlant,
    Parser,
    RevoluteJoint,
    RigidBody,
    SpatialInertia_,
    UnitInertia_,
)
from tqdm import tqdm

import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_payload_id.data import compute_base_param_mapping  # noqa: E402
from robot_payload_id.optimization import solve_inertial_param_sdp  # noqa: E402
from robot_payload_id.utils import (  # noqa: E402
    ArmPlantComponents,
    JointData,
    JointParameters,
    process_joint_data,
)

DEFAULT_JOINT_DATA_PATH = Path("/home/galois/Downloads/trajectory_2")
DEFAULT_FR3_PUSHER_URDF = (
    SCRIPT_DIR / "robot_description" / "fr3_description" / "fr3_pusher.urdf"
)


def resolve_path(path: Path, *, default: Path | None = None) -> Path:
    path = default if path is None else Path(path).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {path}. Tried:\n  {tried}")


def strip_urdf_geometry(urdf_text: str) -> str:
    """Remove visual/collision geometry so Drake can parse FR3 STL-heavy URDFs."""
    return re.sub(
        r"\s*<(visual|collision)(?:\s[^>]*)?>.*?</\1>",
        "",
        urdf_text,
        flags=re.DOTALL,
    )


def arm_joint_names(prefix: str, num_joints: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, num_joints + 1)]


def create_fr3_plant(
    robot_model_path: Path,
    *,
    model_instance_name: str,
    weld_base_link_name: str,
    joint_prefix: str,
    num_arm_joints: int,
    strip_geometry: bool,
) -> ArmPlantComponents:
    builder = DiagramBuilder()
    plant, _scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

    urdf_text = robot_model_path.read_text()
    if strip_geometry:
        urdf_text = strip_urdf_geometry(urdf_text)

    parser = Parser(plant)
    parser.AddModelsFromString(urdf_text, "urdf")
    plant.GetModelInstanceByName(model_instance_name)

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(weld_base_link_name))

    for name in arm_joint_names(joint_prefix, num_arm_joints):
        joint = plant.GetJointByName(name)
        if not isinstance(joint, RevoluteJoint):
            raise TypeError(f"{name} must be a revolute joint, got {type(joint)}.")
        plant.AddJointActuator(name, joint)

    plant.Finalize()

    if plant.num_positions() != num_arm_joints:
        raise RuntimeError(
            f"Expected {num_arm_joints} positions, got {plant.num_positions()}."
        )
    if plant.num_velocities() != num_arm_joints:
        raise RuntimeError(
            f"Expected {num_arm_joints} velocities, got {plant.num_velocities()}."
        )
    if plant.num_actuators() != num_arm_joints:
        raise RuntimeError(
            f"Expected {num_arm_joints} actuators, got {plant.num_actuators()}."
        )

    logging.info(
        "Loaded FR3 model with %d positions, %d velocities, %d actuators.",
        plant.num_positions(),
        plant.num_velocities(),
        plant.num_actuators(),
    )
    logging.info(
        "Actuated arm joints: %s",
        ", ".join(arm_joint_names(joint_prefix, num_arm_joints)),
    )

    return ArmPlantComponents(
        plant=plant,
        plant_context=plant.CreateDefaultContext(),
    )


def body_spatial_inertia_params(
    body: RigidBody,
    context,
) -> tuple[float, np.ndarray, np.ndarray]:
    spatial_inertia = body.CalcSpatialInertiaInBodyFrame(context)
    mass = spatial_inertia.get_mass()
    com = spatial_inertia.get_com()
    rot_inertia = spatial_inertia.CalcRotationalInertia().CopyToFullMatrix3()
    return mass, com, rot_inertia


def combined_welded_subgraph_params(
    plant: MultibodyPlant,
    context,
    representative_body: RigidBody,
) -> tuple[float, np.ndarray, np.ndarray, list[RigidBody]]:
    welded_bodies = [
        body
        for body in plant.GetBodiesWeldedTo(representative_body)
        if body.index() != plant.world_body().index()
    ]

    if len(welded_bodies) == 1:
        mass, com, rot_inertia = body_spatial_inertia_params(welded_bodies[0], context)
    else:
        spatial_inertia = plant.CalcSpatialInertia(
            context=context,
            frame_F=representative_body.body_frame(),
            body_indexes=[body.index() for body in welded_bodies],
        )
        mass = spatial_inertia.get_mass()
        com = spatial_inertia.get_com()
        rot_inertia = spatial_inertia.CalcRotationalInertia().CopyToFullMatrix3()

    return mass, com, rot_inertia, welded_bodies


def make_joint_parameters(
    mass,
    com,
    rot_inertia,
    *,
    reflected_inertia=None,
    viscous_friction=None,
    dynamic_dry_friction=None,
) -> JointParameters:
    return JointParameters(
        m=mass,
        cx=com[0],
        cy=com[1],
        cz=com[2],
        hx=mass * com[0],
        hy=mass * com[1],
        hz=mass * com[2],
        Ixx=rot_inertia[0, 0],
        Iyy=rot_inertia[1, 1],
        Izz=rot_inertia[2, 2],
        Ixy=rot_inertia[0, 1],
        Ixz=rot_inertia[0, 2],
        Iyz=rot_inertia[1, 2],
        reflected_inertia=reflected_inertia,
        viscous_friction=viscous_friction,
        dynamic_dry_friction=dynamic_dry_friction,
    )


def get_fr3_joint_params(
    plant_components: ArmPlantComponents,
    joint_names: Iterable[str],
    *,
    identify_reflected_inertia: bool,
    identify_viscous_friction: bool,
    identify_dynamic_dry_friction: bool,
) -> list[JointParameters]:
    plant = plant_components.plant
    context = plant_components.plant_context

    joint_params: list[JointParameters] = []
    for name in joint_names:
        joint = plant.GetJointByName(name)
        actuator = plant.GetJointActuatorByName(name)
        representative_body = joint.child_body()
        mass, com, rot_inertia, welded_bodies = combined_welded_subgraph_params(
            plant, context, representative_body
        )

        if len(welded_bodies) > 1:
            logging.info(
                "Joint %s uses welded inertia of bodies: %s",
                name,
                ", ".join(body.name() for body in welded_bodies),
            )

        joint_params.append(
            make_joint_parameters(
                mass,
                com,
                rot_inertia,
                reflected_inertia=(
                    actuator.rotor_inertia(context) * actuator.gear_ratio(context) ** 2
                    if identify_reflected_inertia
                    else None
                ),
                viscous_friction=(
                    joint.GetDamping(context) if identify_viscous_friction else None
                ),
                dynamic_dry_friction=(0.0 if identify_dynamic_dry_friction else None),
            )
        )

    return joint_params


def autodiff_value(value: float, num_derivatives: int) -> AutoDiffXd:
    return AutoDiffXd(value, np.zeros(num_derivatives))


def set_zero_spatial_inertia(body: RigidBody, context, num_derivatives: int) -> None:
    zero = autodiff_value(0.0, num_derivatives)
    one = autodiff_value(1.0, num_derivatives)
    body.SetSpatialInertiaInBodyFrame(
        context,
        SpatialInertia_[AutoDiffXd](
            zero,
            [zero, zero, zero],
            UnitInertia_[AutoDiffXd](one, one, one, zero, zero, zero),
            skip_validity_check=True,
        ),
    )


def create_fr3_autodiff_plant(
    plant_components: ArmPlantComponents,
    joint_names: list[str],
    params_initial: list[JointParameters],
    *,
    identify_reflected_inertia: bool,
    identify_viscous_friction: bool,
    identify_dynamic_dry_friction: bool,
) -> ArmPlantComponents:
    num_joints = len(joint_names)
    num_params_per_joint = (
        10
        + int(identify_reflected_inertia)
        + int(identify_viscous_friction)
        + int(identify_dynamic_dry_friction)
    )
    num_params = num_joints * num_params_per_joint

    ad_plant: MultibodyPlant = plant_components.plant.ToAutoDiffXd()
    ad_context = ad_plant.CreateDefaultContext()
    ad_context.SetTimeStateAndParametersFrom(plant_components.plant_context)

    ad_parameters: list[JointParameters] = []
    for i, (name, params) in enumerate(zip(joint_names, params_initial)):
        offset = i * num_params_per_joint

        def make_ad(value: float, param_offset: int) -> AutoDiffXd:
            derivatives = np.zeros(num_params)
            derivatives[offset + param_offset] = 1.0
            return AutoDiffXd(float(value), derivatives)

        m_ad = make_ad(params.m, 0)
        hx_ad = make_ad(params.hx, 1)
        hy_ad = make_ad(params.hy, 2)
        hz_ad = make_ad(params.hz, 3)
        cx_ad = hx_ad / m_ad
        cy_ad = hy_ad / m_ad
        cz_ad = hz_ad / m_ad

        Ixx_ad = make_ad(params.Ixx, 4)
        Ixy_ad = make_ad(params.Ixy, 5)
        Ixz_ad = make_ad(params.Ixz, 6)
        Iyy_ad = make_ad(params.Iyy, 7)
        Iyz_ad = make_ad(params.Iyz, 8)
        Izz_ad = make_ad(params.Izz, 9)

        param_offset = 10
        reflected_inertia_ad = None
        if identify_reflected_inertia:
            reflected_inertia_ad = make_ad(
                params.reflected_inertia or 0.0, param_offset
            )
            param_offset += 1

        viscous_friction_ad = None
        if identify_viscous_friction:
            viscous_friction_ad = make_ad(params.viscous_friction or 0.0, param_offset)
            param_offset += 1

        dynamic_dry_friction_ad = None
        if identify_dynamic_dry_friction:
            dynamic_dry_friction_ad = make_ad(
                params.dynamic_dry_friction or 0.0, param_offset
            )

        ad_joint = ad_plant.GetJointByName(name)
        ad_actuator = ad_plant.GetJointActuatorByName(name)
        representative_body = ad_joint.child_body()
        welded_bodies = [
            body
            for body in ad_plant.GetBodiesWeldedTo(representative_body)
            if body.index() != ad_plant.world_body().index()
        ]

        # Represent the whole welded subgraph by the arm-link child body so the
        # parameter vector is still one 10D inertial block per arm joint.
        for body in welded_bodies:
            if body.index() != representative_body.index():
                set_zero_spatial_inertia(body, ad_context, num_params)

        spatial_inertia_ad = SpatialInertia_[AutoDiffXd](
            m_ad,
            [cx_ad, cy_ad, cz_ad],
            UnitInertia_[AutoDiffXd](
                Ixx_ad / m_ad,
                Iyy_ad / m_ad,
                Izz_ad / m_ad,
                Ixy_ad / m_ad,
                Ixz_ad / m_ad,
                Iyz_ad / m_ad,
            ),
            skip_validity_check=True,
        )
        representative_body.SetSpatialInertiaInBodyFrame(ad_context, spatial_inertia_ad)

        if identify_reflected_inertia:
            ad_actuator.SetRotorInertia(ad_context, 0.0)
        if identify_viscous_friction:
            ad_joint.SetDamping(ad_context, viscous_friction_ad)

        ad_parameters.append(
            JointParameters(
                m=m_ad,
                cx=cx_ad,
                cy=cy_ad,
                cz=cz_ad,
                hx=hx_ad,
                hy=hy_ad,
                hz=hz_ad,
                Ixx=Ixx_ad,
                Ixy=Ixy_ad,
                Ixz=Ixz_ad,
                Iyy=Iyy_ad,
                Iyz=Iyz_ad,
                Izz=Izz_ad,
                reflected_inertia=reflected_inertia_ad,
                viscous_friction=viscous_friction_ad,
                dynamic_dry_friction=dynamic_dry_friction_ad,
            )
        )

    return ArmPlantComponents(
        plant=ad_plant,
        plant_context=ad_context,
        parameters=ad_parameters,
    )


def extract_fr3_data_matrix_autodiff(
    plant_components: ArmPlantComponents,
    joint_data: JointData,
    joint_names: list[str],
    params_initial: list[JointParameters],
    *,
    identify_reflected_inertia: bool,
    identify_viscous_friction: bool,
    identify_dynamic_dry_friction: bool,
    use_progress_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ad_components = create_fr3_autodiff_plant(
        plant_components,
        joint_names,
        params_initial,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
    )

    num_timesteps = len(joint_data.sample_times_s)
    num_joints = len(joint_names)
    ad_params = np.concatenate(
        [params.get_lumped_param_list() for params in ad_components.parameters]
    )
    param_values = np.array([param.value() for param in ad_params])
    num_lumped_params = len(ad_params)

    W_data = np.zeros((num_timesteps * num_joints, num_lumped_params))
    tau_model = np.zeros(num_timesteps * num_joints)
    w0_data = np.zeros(num_timesteps * num_joints)

    reflected_inertia_ad = None
    if identify_reflected_inertia:
        reflected_inertia_ad = np.array(
            [params.reflected_inertia for params in ad_components.parameters]
        )

    dynamic_dry_friction_ad = None
    if identify_dynamic_dry_friction:
        dynamic_dry_friction_ad = np.array(
            [params.dynamic_dry_friction for params in ad_components.parameters]
        )

    for i in tqdm(
        range(num_timesteps),
        desc="Extracting FR3 data matrix",
        disable=not use_progress_bar,
    ):
        ad_components.plant.SetPositions(
            ad_components.plant_context, joint_data.joint_positions[i]
        )
        ad_components.plant.SetVelocities(
            ad_components.plant_context, joint_data.joint_velocities[i]
        )

        forces = MultibodyForces_[AutoDiffXd](ad_components.plant)
        ad_components.plant.CalcForceElementsContribution(
            ad_components.plant_context, forces
        )
        ad_torques = ad_components.plant.CalcInverseDynamics(
            context=ad_components.plant_context,
            known_vdot=joint_data.joint_accelerations[i],
            external_forces=forces,
        )

        if identify_reflected_inertia:
            ad_torques += reflected_inertia_ad * joint_data.joint_accelerations[i]
        if identify_dynamic_dry_friction:
            dry_torque = dynamic_dry_friction_ad * np.sign(
                joint_data.joint_velocities[i]
            )
            dry_torque[np.abs(joint_data.joint_velocities[i]) < 0.001] = 0.0
            ad_torques += dry_torque

        row = slice(i * num_joints, (i + 1) * num_joints)
        W_data[row, :] = np.vstack([torque.derivatives() for torque in ad_torques])
        tau_model[row] = np.array([torque.value() for torque in ad_torques])
        w0_data[row] = tau_model[row] - W_data[row, :] @ param_values

    return W_data, w0_data, tau_model


def validate_joint_data(joint_data: JointData) -> None:
    arrays = {
        "joint_positions": joint_data.joint_positions,
        "joint_velocities": joint_data.joint_velocities,
        "joint_accelerations": joint_data.joint_accelerations,
        "joint_torques": joint_data.joint_torques,
        "sample_times_s": joint_data.sample_times_s,
    }
    for name, values in arrays.items():
        if values is None:
            raise ValueError(f"{name} is missing.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values.")


def project_to_base_parameters(
    W_data_raw: np.ndarray,
    joint_data: JointData,
    plant_components: ArmPlantComponents,
    joint_names: list[str],
    params_initial: list[JointParameters],
    *,
    identify_reflected_inertia: bool,
    identify_viscous_friction: bool,
    identify_dynamic_dry_friction: bool,
    keep_unidentifiable_params: bool,
    base_param_mapping_path: Path | None,
    num_mapping_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if keep_unidentifiable_params:
        return W_data_raw, None

    base_param_mapping = None
    if base_param_mapping_path is not None:
        logging.info("Loading base parameter mapping from %s", base_param_mapping_path)
        base_param_mapping = np.load(base_param_mapping_path)

    if base_param_mapping is None or W_data_raw.shape[1] != base_param_mapping.shape[0]:
        logging.warning(
            "Base parameter mapping not provided or has wrong shape. Recomputing."
        )
        if len(joint_data.sample_times_s) > num_mapping_samples:
            rng = np.random.default_rng(0)
            random_joint_data = JointData(
                joint_positions=rng.random((num_mapping_samples, len(joint_names)))
                - 0.5,
                joint_velocities=rng.random((num_mapping_samples, len(joint_names)))
                - 0.5,
                joint_accelerations=rng.random((num_mapping_samples, len(joint_names)))
                - 0.5,
                joint_torques=np.zeros((num_mapping_samples, len(joint_names))),
                sample_times_s=np.zeros(num_mapping_samples),
            )
            W_for_mapping, _, _ = extract_fr3_data_matrix_autodiff(
                plant_components,
                random_joint_data,
                joint_names,
                params_initial,
                identify_reflected_inertia=identify_reflected_inertia,
                identify_viscous_friction=identify_viscous_friction,
                identify_dynamic_dry_friction=identify_dynamic_dry_friction,
                use_progress_bar=False,
            )
        else:
            W_for_mapping = W_data_raw
        base_param_mapping = compute_base_param_mapping(W_for_mapping)

    logging.info(
        "%d out of %d parameters are identifiable.",
        base_param_mapping.shape[1],
        base_param_mapping.shape[0],
    )

    if base_param_mapping.shape[0] == base_param_mapping.shape[1]:
        logging.warning("All parameters are identifiable. Not applying projection.")
        return W_data_raw, None

    return W_data_raw @ base_param_mapping, base_param_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve SDP inertial system ID for recorded FR3 pusher joint data."
    )
    parser.add_argument(
        "--joint_data_path",
        type=Path,
        default=DEFAULT_JOINT_DATA_PATH,
        help=f"JointData directory. Default: {DEFAULT_JOINT_DATA_PATH}",
    )
    parser.add_argument(
        "--robot_model_path",
        type=Path,
        default=DEFAULT_FR3_PUSHER_URDF,
        help=f"FR3 URDF path. Default: {DEFAULT_FR3_PUSHER_URDF}",
    )
    parser.add_argument("--model_instance_name", default="panda")
    parser.add_argument("--weld_base_link_name", default="panda_link0")
    parser.add_argument("--joint_prefix", default="panda_joint")
    parser.add_argument("--num_arm_joints", type=int, default=7)
    parser.add_argument(
        "--keep_geometry",
        action="store_true",
        help="Do not strip visual/collision tags before Drake parsing.",
    )
    parser.add_argument("--output_param_path", type=Path)
    parser.add_argument("--base_param_mapping_path", type=Path)
    parser.add_argument("--keep_unidentifiable_params", action="store_true")
    parser.add_argument("--num_mapping_samples", type=int, default=2000)
    parser.add_argument("--kPrintToConsole", action="store_true")
    parser.add_argument(
        "--not_identify_reflected_inertia",
        action="store_true",
        help="Do not identify reflected inertia.",
    )
    parser.add_argument(
        "--not_identify_viscous_friction",
        action="store_true",
        help="Do not identify viscous friction.",
    )
    parser.add_argument(
        "--not_identify_dynamic_dry_friction",
        action="store_true",
        help="Do not identify dynamic dry friction.",
    )
    parser.add_argument("--use_euclidean_regularization", action="store_true")
    parser.add_argument("--regularization_weight", type=float, default=1e-2)
    parser.add_argument("--known_max_mass", type=float)
    parser.add_argument(
        "--process_joint_data",
        action="store_true",
        help="Filter data and recompute velocities/accelerations.",
    )
    parser.add_argument("--num_endpoints_to_remove", type=int, default=1)
    parser.add_argument("--not_compute_velocities", action="store_true")
    parser.add_argument("--not_filter_positions", action="store_true")
    parser.add_argument("--pos_order", type=int, default=10)
    parser.add_argument("--pos_cutoff_freq_hz", type=float, default=60.0)
    parser.add_argument("--vel_order", type=int, default=10)
    parser.add_argument("--vel_cutoff_freq_hz", type=float, default=5.6)
    parser.add_argument("--acc_order", type=int, default=10)
    parser.add_argument("--acc_cutoff_freq_hz", type=float, default=4.2)
    parser.add_argument("--torque_order", type=int, default=10)
    parser.add_argument("--torque_cutoff_freq_hz", type=float, default=4.0)
    parser.add_argument(
        "--wandb_mode",
        choices=["disabled", "online", "offline"],
        default="disabled",
    )
    parser.add_argument(
        "--log_level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    joint_data_path = resolve_path(args.joint_data_path)
    robot_model_path = resolve_path(args.robot_model_path)
    base_param_mapping_path = (
        resolve_path(args.base_param_mapping_path)
        if args.base_param_mapping_path is not None
        else None
    )

    identify_reflected_inertia = not args.not_identify_reflected_inertia
    identify_viscous_friction = not args.not_identify_viscous_friction
    identify_dynamic_dry_friction = not args.not_identify_dynamic_dry_friction
    joint_names = arm_joint_names(args.joint_prefix, args.num_arm_joints)

    wandb.init(
        project="robot_payload_id",
        name="fr3 inertial_param_sdp",
        config=vars(args),
        mode=args.wandb_mode,
    )

    logging.info("Loading joint data from %s", joint_data_path)
    joint_data = JointData.load_from_disk_allow_missing(joint_data_path)
    if args.process_joint_data:
        joint_data = process_joint_data(
            joint_data=joint_data,
            num_endpoints_to_remove=args.num_endpoints_to_remove,
            compute_velocities=not args.not_compute_velocities,
            filter_positions=not args.not_filter_positions,
            pos_filter_order=args.pos_order,
            pos_cutoff_freq_hz=args.pos_cutoff_freq_hz,
            vel_filter_order=args.vel_order,
            vel_cutoff_freq_hz=args.vel_cutoff_freq_hz,
            acc_filter_order=args.acc_order,
            acc_cutoff_freq_hz=args.acc_cutoff_freq_hz,
            torque_filter_order=args.torque_order,
            torque_cutoff_freq_hz=args.torque_cutoff_freq_hz,
        )
    elif joint_data.joint_accelerations is None or not np.all(
        np.isfinite(joint_data.joint_accelerations)
    ):
        raise ValueError(
            "Joint accelerations are missing or non-finite. Re-run with "
            "--process_joint_data."
        )

    validate_joint_data(joint_data)
    logging.info(
        "Using %d samples over %.3f seconds.",
        len(joint_data.sample_times_s),
        joint_data.sample_times_s[-1] - joint_data.sample_times_s[0],
    )

    plant_components = create_fr3_plant(
        robot_model_path,
        model_instance_name=args.model_instance_name,
        weld_base_link_name=args.weld_base_link_name,
        joint_prefix=args.joint_prefix,
        num_arm_joints=args.num_arm_joints,
        strip_geometry=not args.keep_geometry,
    )
    params_initial = get_fr3_joint_params(
        plant_components,
        joint_names,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
    )

    W_data_raw, w0_data, _ = extract_fr3_data_matrix_autodiff(
        plant_components,
        joint_data,
        joint_names,
        params_initial,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
    )
    tau_data = joint_data.joint_torques.flatten() - w0_data

    W_data, base_param_mapping = project_to_base_parameters(
        W_data_raw,
        joint_data,
        plant_components,
        joint_names,
        params_initial,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
        keep_unidentifiable_params=args.keep_unidentifiable_params,
        base_param_mapping_path=base_param_mapping_path,
        num_mapping_samples=args.num_mapping_samples,
    )

    _, result, variable_names, variable_vec, _ = solve_inertial_param_sdp(
        num_links=args.num_arm_joints,
        W_data=W_data,
        tau_data=tau_data,
        base_param_mapping=base_param_mapping,
        regularization_weight=args.regularization_weight,
        params_guess=params_initial,
        use_euclidean_regularization=args.use_euclidean_regularization,
        identify_reflected_inertia=identify_reflected_inertia,
        identify_viscous_friction=identify_viscous_friction,
        identify_dynamic_dry_friction=identify_dynamic_dry_friction,
        known_max_mass=args.known_max_mass,
        solver_kPrintToConsole=args.kPrintToConsole,
    )

    if not result.is_success():
        wandb.log({"sdp_cost": np.inf})
        logging.warning("Failed to solve FR3 inertial parameter SDP.")
        logging.info("Solution result:\n%s", result.get_solution_result())
        logging.info("Solver details:\n%s", result.get_solver_details())
        return

    final_cost = result.get_optimal_cost()
    wandb.log({"sdp_cost": final_cost})
    logging.info("Final cost: %s", final_cost)
    var_sol_dict = dict(zip(variable_names, result.GetSolution(variable_vec)))
    logging.info("SDP result:\n%s", var_sol_dict)

    if args.output_param_path is not None:
        output_param_path = (
            args.output_param_path
            if args.output_param_path.is_absolute()
            else PROJECT_ROOT / args.output_param_path
        )
        output_param_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_param_path, var_sol_dict)
        logging.info("Saved parameters to %s", output_param_path)


if __name__ == "__main__":
    main()
