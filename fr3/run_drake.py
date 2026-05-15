import argparse
import os
import select
import sys
import time
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from loguru import logger
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    Rgba,
    RigidTransform,
    SpatialInertia,
    Sphere,
    StartMeshcat,
    UnitInertia,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from system_identification.drake.camera_collision import (
    CAMERA_BOX_HEIGHT_SCALE,
    CAMERA_BOX_MARGIN_SCALE,
    CAMERA_BOX_SPECS_MM,
    DEFAULT_FIXED_COLLISION_JOINTS,
    DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
    DrakeCameraCollisionChecker,
    find_robot_urdf,
    normalize_robot_body_pairs,
)

from robot_payload_id.data import (
    compute_autodiff_joint_data_from_fourier_series_traj_params1,
)
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes

DEFAULT_FR3_URDF = (
    SCRIPT_DIR / "robot_description" / "fr3_description" / "fr3_gripper.urdf"
)
DEFAULT_TRAJECTORY_ROOT = SCRIPT_DIR / "logs"


def _candidate_paths(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return [path]
    return [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        SCRIPT_DIR / path,
        DEFAULT_TRAJECTORY_ROOT / path,
        DEFAULT_TRAJECTORY_ROOT / "fr3_excitation" / path,
    ]


def resolve_traj_path(path):
    if path is not None:
        candidates = _candidate_paths(path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
            sibling_csv = candidate.with_suffix(".csv")
            if sibling_csv.exists():
                return sibling_csv.resolve()
        candidates_str = "\n  ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Could not find trajectory '{path}'. Tried:\n  {candidates_str}"
        )

    candidates = []
    folder = DEFAULT_TRAJECTORY_ROOT
    if folder.exists():
        for pattern in (
            "**/selected.csv",
            "**/best_condition.csv",
            "**/final.csv",
            "**/initial.csv",
            "**/*.npy",
        ):
            candidates.extend(folder.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No saved trajectory found under {folder}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def load_trajectory(path, *, num_timesteps, time_horizon):
    if path.is_dir():
        folder_final_csv = path / "final.csv"
        sibling_final_csv = path.with_suffix(".csv")
        if folder_final_csv.exists():
            return load_trajectory(
                folder_final_csv,
                num_timesteps=num_timesteps,
                time_horizon=time_horizon,
            )
        if sibling_final_csv.exists():
            return load_trajectory(
                sibling_final_csv,
                num_timesteps=num_timesteps,
                time_horizon=time_horizon,
            )
        if (path / "a_value.npy").exists():
            traj_attrs = FourierSeriesTrajectoryAttributes.load(path)
            joint_data = compute_autodiff_joint_data_from_fourier_series_traj_params1(
                num_timesteps=num_timesteps,
                time_horizon=time_horizon,
                traj_attrs=traj_attrs,
                use_progress_bar=False,
            )
            return (
                joint_data.sample_times_s,
                np.asarray(joint_data.joint_positions, dtype=float),
                np.asarray(joint_data.joint_velocities, dtype=float),
                np.asarray(joint_data.joint_accelerations, dtype=float),
            )
        raise ValueError(
            f"Directory trajectory {path} must contain final.csv or Fourier "
            "files a_value.npy, b_value.npy, q0_value.npy, omega.npy."
        )

    if path.suffix == ".npy":
        traj = np.load(path, allow_pickle=True).item()
        return traj["t"], traj["q"], traj["dq"], traj["ddq"]

    if path.suffix == ".csv":
        raw = np.genfromtxt(path, delimiter=",", names=True)
        columns = list(raw.dtype.names)
        t = np.asarray(raw["t"], dtype=float)
        q_cols = sorted(
            [name for name in columns if name.startswith("q_")],
            key=lambda name: int(name.split("_")[1]),
        )
        dq_cols = sorted(
            [name for name in columns if name.startswith("dq_")],
            key=lambda name: int(name.split("_")[1]),
        )
        ddq_cols = sorted(
            [name for name in columns if name.startswith("ddq_")],
            key=lambda name: int(name.split("_")[1]),
        )
        if not q_cols or not dq_cols or not ddq_cols:
            raise ValueError(f"CSV trajectory {path} is missing q/dq/ddq columns")

        q = np.column_stack([raw[name] for name in q_cols]).astype(float)
        dq = np.column_stack([raw[name] for name in dq_cols]).astype(float)
        ddq = np.column_stack([raw[name] for name in ddq_cols]).astype(float)
        return t, q, dq, ddq

    raise ValueError(f"Unsupported trajectory file format: {path.suffix}")


def resolve_robot_urdf_path(robot_name, robot_urdf_path):
    if robot_urdf_path is not None:
        return Path(robot_urdf_path).expanduser().resolve()
    if robot_name == "fr3" and DEFAULT_FR3_URDF.exists():
        return DEFAULT_FR3_URDF.resolve()
    return find_robot_urdf(robot_name)


def parse_urdf_tolerant(urdf_path):
    try:
        return ET.parse(urdf_path)
    except ET.ParseError as parse_error:
        lines = Path(urdf_path).read_text(encoding="utf-8").splitlines()
        cleaned_lines = []
        previous_nonempty = ""
        removed_extra_link_close = False
        for line in lines:
            stripped = line.strip()
            if (
                not removed_extra_link_close
                and stripped == "</link>"
                and previous_nonempty == "</joint>"
            ):
                removed_extra_link_close = True
                continue
            cleaned_lines.append(line)
            if stripped:
                previous_nonempty = stripped
        if not removed_extra_link_close:
            raise parse_error
        return ET.ElementTree(ET.fromstring("\n".join(cleaned_lines) + "\n"))


def fix_visualization_only_joints(
    urdf_root,
    joint_names=DEFAULT_FIXED_COLLISION_JOINTS,
):
    joint_names = set(joint_names)
    for joint in urdf_root.findall("joint"):
        if joint.attrib.get("name") not in joint_names:
            continue
        joint.attrib["type"] = "fixed"
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for elem in list(joint.findall(tag)):
                joint.remove(elem)


def root_link_name(urdf_root):
    link_names = {link.attrib["name"] for link in urdf_root.findall("link")}
    child_links = {
        joint.find("child").attrib["link"]
        for joint in urdf_root.findall("joint")
        if joint.find("child") is not None
    }
    roots = [name for name in link_names - child_links if name != "world"]
    if len(roots) != 1:
        raise ValueError(f"Expected one non-world root link, found {roots}")
    return roots[0]


def visual_only_urdf_xml(urdf_path):
    tree = parse_urdf_tolerant(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        for elem in list(link.findall("collision")):
            link.remove(elem)

    for joint in root.findall("joint"):
        for elem in list(joint.findall("safety_controller")):
            joint.remove(elem)

    fix_visualization_only_joints(root)

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.set("filename", (urdf_path.parent / filename).resolve().as_uri())

    return root, ET.tostring(root, encoding="unicode")


def camera_box_specs(xy_prism_height, *, xy_scale, z_scale):
    boxes = []
    colors = (
        np.array([0.85, 0.2, 0.2, 0.28]),
        np.array([0.2, 0.45, 0.9, 0.28]),
        np.array([0.85, 0.2, 0.2, 0.28]),
        np.array([0.2, 0.45, 0.9, 0.28]),
        np.array([0.2, 0.75, 0.35, 0.28]),
    )
    for idx, (name, center_mm, size_mm) in enumerate(CAMERA_BOX_SPECS_MM):
        color = colors[idx % len(colors)]
        center = center_mm / 1000.0
        size = size_mm.astype(float) / 1000.0
        size[:2] *= float(xy_scale)
        if xy_prism_height is not None:
            size[2] = float(xy_prism_height) * float(z_scale)
        else:
            size[2] *= float(z_scale)
        boxes.append((name, center, size, color))
    return boxes


def register_camera_visual_geometry(plant, body, name, size, color, chamfer_radius):
    chamfer = min(max(chamfer_radius, 0.0), 0.49 * min(size[0], size[1]))
    if chamfer <= 0.0:
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(),
            Box(*size),
            f"{name}_visual",
            color,
        )
        return

    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Box(max(size[0] - 2.0 * chamfer, 1e-6), size[1], size[2]),
        f"{name}_x_strip_visual",
        color,
    )
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        Box(size[0], max(size[1] - 2.0 * chamfer, 1e-6), size[2]),
        f"{name}_y_strip_visual",
        color,
    )

    corner_shape = Cylinder(chamfer, size[2])
    corner_offsets = (
        np.array([size[0] / 2.0 - chamfer, size[1] / 2.0 - chamfer, 0.0]),
        np.array([size[0] / 2.0 - chamfer, -size[1] / 2.0 + chamfer, 0.0]),
        np.array([-size[0] / 2.0 + chamfer, size[1] / 2.0 - chamfer, 0.0]),
        np.array([-size[0] / 2.0 + chamfer, -size[1] / 2.0 + chamfer, 0.0]),
    )
    for idx, offset in enumerate(corner_offsets):
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(offset),
            corner_shape,
            f"{name}_corner_{idx}_visual",
            color,
        )


def build_visualization_diagram(
    robot_name,
    meshcat,
    *,
    robot_urdf_path,
    camera_chamfer_radius,
    xy_prism_height,
    camera_box_xy_scale,
    camera_box_z_scale,
):
    urdf_path = resolve_robot_urdf_path(robot_name, robot_urdf_path)
    urdf_root, urdf_xml = visual_only_urdf_xml(urdf_path)
    root_link = root_link_name(urdf_root)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    model_instance = Parser(plant).AddModelsFromString(urdf_xml, "urdf")[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetBodyByName(root_link, model_instance).body_frame(),
    )

    camera_model_instance = plant.AddModelInstance("camera_boxes_visual")
    inertia = SpatialInertia(
        mass=1.0,
        p_PScm_E=np.zeros(3),
        G_SP_E=UnitInertia.SolidBox(1.0, 1.0, 1.0),
    )
    for name, center, size, color in camera_box_specs(
        xy_prism_height,
        xy_scale=camera_box_xy_scale,
        z_scale=camera_box_z_scale,
    ):
        body = plant.AddRigidBody(name, camera_model_instance, inertia)
        plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(center))
        register_camera_visual_geometry(
            plant,
            body,
            name,
            size,
            color,
            camera_chamfer_radius,
        )

    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    return diagram, context, plant, plant_context, model_instance


def report_drake_collision(
    q,
    t,
    *,
    robot_name,
    robot_urdf_path,
    stride,
    min_distance,
    robot_sphere_radius,
    robot_link_samples,
    camera_chamfer_radius,
    xy_prism_height,
    camera_box_xy_scale,
    camera_box_z_scale,
):
    checker = DrakeCameraCollisionChecker(
        robot_name=robot_name,
        robot_urdf_path=resolve_robot_urdf_path(robot_name, robot_urdf_path),
        min_distance=min_distance,
        robot_sphere_radius=robot_sphere_radius,
        robot_link_samples=robot_link_samples,
        camera_chamfer_radius=camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
        camera_box_xy_scale=camera_box_xy_scale,
        camera_box_z_scale=camera_box_z_scale,
    )
    sample_indices = np.arange(0, len(q), max(1, int(stride)))
    values = checker.constraint_values(q[sample_indices])
    min_idx = int(np.argmin(values))
    min_margin = float(values[min_idx])
    global_idx = int(sample_indices[min_idx])
    logger.info(
        "Drake camera collision margin: "
        f"min={min_margin} at sample={global_idx}, t={t[global_idx]}"
    )
    if min_margin < 0.0:
        logger.warning("Trajectory violates the Drake camera collision constraint.")
    else:
        logger.info(
            "Trajectory satisfies the sampled Drake camera collision constraint."
        )
    return checker, values


def report_link_y_bounds(q, t, checker, *, stride, lower, upper):
    sample_indices = np.arange(0, len(q), max(1, int(stride)))
    margins = checker.robot_link_y_margins(q[sample_indices], lower=lower, upper=upper)
    lower_idx = int(np.argmin(margins[:, 0]))
    upper_idx = int(np.argmin(margins[:, 1]))
    lower_global_idx = int(sample_indices[lower_idx])
    upper_global_idx = int(sample_indices[upper_idx])
    min_margin = float(np.min(margins))
    logger.info(
        f"Drake link-y lower margin: min={margins[lower_idx, 0]} "
        f"at sample={lower_global_idx}, t={t[lower_global_idx]}"
    )
    logger.info(
        f"Drake link-y upper margin: min={margins[upper_idx, 1]} "
        f"at sample={upper_global_idx}, t={t[upper_global_idx]}"
    )
    if min_margin < 0.0:
        logger.warning(f"Trajectory violates link-y bounds: {lower} < y < {upper}.")
    else:
        logger.info(
            f"Trajectory satisfies sampled link-y bounds: {lower} < y < {upper}."
        )
    return margins


def robot_link_workspace_margins(
    checker,
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
            checker,
            q_i,
            min_link_index=1,
        )
        z_points, z_radii = robot_workspace_sample_points_and_radii(
            checker,
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


def robot_workspace_sample_points(checker, q_i, *, min_link_index):
    points, _radii = robot_workspace_sample_points_and_radii(
        checker,
        q_i,
        min_link_index=min_link_index,
    )
    return points


def robot_workspace_sample_points_and_radii(checker, q_i, *, min_link_index):
    if not hasattr(checker, "_robot_sample_specs"):
        points = checker._robot_sample_points(q_i)
        radii = np.full(len(points), checker.robot_sphere_radius)
        return points, radii

    checker.plant.SetPositions(checker.plant_context, checker.model_instance, q_i)
    points = []
    radii = []
    for body, offset, radius in checker._robot_sample_specs:
        body_name = body.name()
        if not robot_body_at_or_after_link(body_name, min_link_index):
            continue
        x_wb = checker.plant.EvalBodyPoseInWorld(checker.plant_context, body)
        points.append(x_wb.multiply(offset))
        radii.append(float(radius))
    if not points:
        points = checker._robot_sample_points(q_i)
        radii = np.full(len(points), checker.robot_sphere_radius)
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


def report_robot_self_collision(
    q,
    t,
    checker,
    *,
    stride,
    clearance,
    body_pairs=DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
):
    sample_indices = np.arange(0, len(q), max(1, int(stride)))
    margins = checker.robot_self_collision_pair_margins(
        q[sample_indices],
        body_pairs=body_pairs,
    )
    if margins.size == 0:
        logger.info("Franka self-collision check skipped: no configured body pairs.")
        return np.inf

    local_sample_idx, pair_idx = np.unravel_index(
        int(np.argmin(margins)), margins.shape
    )
    best_margin = float(margins[local_sample_idx, pair_idx])
    best_global_idx = int(sample_indices[local_sample_idx])
    first_body, second_body = normalize_robot_body_pairs(body_pairs)[pair_idx]
    logger.info(
        "Franka self-collision sphere margin: "
        f"min={best_margin} at sample={best_global_idx}, t={t[best_global_idx]}, "
        f"{first_body} vs {second_body} "
        f"(clearance threshold={clearance})"
    )
    if best_margin < float(clearance):
        logger.warning(
            "Possible Franka self-collision under saved-sphere approximation: "
            f"margin {best_margin} < {clearance}."
        )
    else:
        logger.info("Trajectory satisfies sampled Franka self-collision sphere check.")
    return best_margin


def report_link_workspace_bounds(
    q,
    t,
    checker,
    *,
    stride,
    x_lower,
    y_lower,
    y_upper,
    z_lower,
):
    sample_indices = np.arange(0, len(q), max(1, int(stride)))
    margins = robot_link_workspace_margins(
        checker,
        q[sample_indices],
        x_lower=x_lower,
        y_lower=y_lower,
        y_upper=y_upper,
        z_lower=z_lower,
    )
    labels = (
        ("x lower", 0, f"x > {x_lower}"),
        ("y lower", 1, f"y > {y_lower}"),
        ("y upper", 2, f"y < {y_upper}"),
        ("z lower", 3, f"z > {z_lower}"),
    )
    for label, column, bound_text in labels:
        local_idx = int(np.argmin(margins[:, column]))
        global_idx = int(sample_indices[local_idx])
        logger.info(
            f"Drake link-{label} margin: min={margins[local_idx, column]} "
            f"at sample={global_idx}, t={t[global_idx]} ({bound_text})"
        )

    if float(np.min(margins)) < 0.0:
        logger.warning(
            "Trajectory violates sampled workspace bounds: "
            f"x > {x_lower}, {y_lower} < y < {y_upper}, z > {z_lower}."
        )
    else:
        logger.info(
            "Trajectory satisfies sampled workspace bounds: "
            f"x > {x_lower}, {y_lower} < y < {y_upper}, z > {z_lower}."
        )
    return margins


def _safe_meshcat_name(name):
    return "".join(char if char.isalnum() or char in "_-" else "_" for char in name)


def robot_sample_points_with_body_names(checker, q_i):
    if not hasattr(checker, "_robot_sample_specs"):
        points = checker._robot_sample_points(q_i)
        return [
            (f"sample_{idx}", point, checker.robot_sphere_radius)
            for idx, point in enumerate(points)
        ]

    checker.plant.SetPositions(checker.plant_context, checker.model_instance, q_i)
    samples = []
    for body, offset, radius in checker._robot_sample_specs:
        x_wb = checker.plant.EvalBodyPoseInWorld(checker.plant_context, body)
        samples.append((body.name(), x_wb.multiply(offset), float(radius)))
    return samples


def setup_robot_sample_markers(meshcat, checker, *, marker_radius):
    meshcat.Delete("/robot_sample_specs")
    for idx, (body, _offset, radius) in enumerate(checker._robot_sample_specs):
        body_name = body.name()
        is_gripper = "hand" in body_name or "finger" in body_name
        color = Rgba(1.0, 0.45, 0.05, 0.95) if is_gripper else Rgba(0.0, 0.55, 1.0, 0.9)
        path = f"/robot_sample_specs/{idx:03d}_{_safe_meshcat_name(body_name)}"
        display_radius = (
            float(radius) if marker_radius is None else float(marker_radius)
        )
        meshcat.SetObject(path, Sphere(max(display_radius, 1e-4)), color)
    logger.info(
        "Visualizing %d robot sample specs in Meshcat: blue=arm, orange=gripper.",
        len(checker._robot_sample_specs),
    )


def publish_robot_sample_markers(meshcat, checker, q_i):
    for idx, (body_name, point, _radius) in enumerate(
        robot_sample_points_with_body_names(checker, q_i)
    ):
        path = f"/robot_sample_specs/{idx:03d}_{_safe_meshcat_name(body_name)}"
        meshcat.SetTransform(path, RigidTransform(point))


def play_trajectory(
    diagram,
    context,
    plant,
    plant_context,
    model_instance,
    t,
    q,
    *,
    playback_stride,
    speed,
    meshcat=None,
    robot_sample_checker=None,
):
    playback_stride = max(1, int(playback_stride))
    speed = max(float(speed), 1e-9)
    indices = list(range(0, len(t), playback_stride))
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)

    for current_idx, next_idx in zip(indices, indices[1:] + [indices[-1]]):
        context.SetTime(float(t[current_idx]))
        plant.SetPositions(plant_context, model_instance, q[current_idx])
        diagram.ForcedPublish(context)
        if meshcat is not None and robot_sample_checker is not None:
            publish_robot_sample_markers(meshcat, robot_sample_checker, q[current_idx])

        if next_idx != current_idx:
            delay = max(float(t[next_idx] - t[current_idx]) / speed, 0.0)
            if delay > 0.0:
                time.sleep(delay)


def wait_for_meshcat_start(meshcat, button_name):
    meshcat.AddButton(button_name, "Space")
    logger.info(
        f"Waiting for Meshcat button '{button_name}' before playback "
        "(or press Space in the Meshcat browser)."
    )
    try:
        while meshcat.GetButtonClicks(button_name) < 1:
            time.sleep(0.1)
    finally:
        meshcat.DeleteButton(button_name, strict=False)


def hold_with_replay_button(
    meshcat,
    button_name,
    play_once,
):
    meshcat.AddButton(button_name, "r")
    logger.info(
        f"Playback finished. Click Meshcat button '{button_name}' "
        "or press r in the Meshcat browser to replay. Press Enter here to exit."
    )
    clicks = meshcat.GetButtonClicks(button_name)
    try:
        while True:
            if sys.stdin.isatty():
                readable, _writable, _error = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    sys.stdin.readline()
                    return
            else:
                return

            new_clicks = meshcat.GetButtonClicks(button_name)
            if new_clicks <= clicks:
                continue

            clicks = new_clicks
            logger.info("Replay requested from Meshcat.")
            play_once()
            logger.info(
                f"Replay finished. Click Meshcat button '{button_name}' again "
                "or press Enter here to exit."
            )
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting Meshcat playback.")
    finally:
        meshcat.DeleteButton(button_name, strict=False)


def main():
    parser = argparse.ArgumentParser(
        description="Replay a trajectory in Drake/Meshcat with camera collision boxes."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=None,
        help="Trajectory CSV/NPY. Defaults to the latest trajectory under fr3/logs.",
    )
    parser.add_argument("--robot", type=str, default="fr3")
    parser.add_argument(
        "--robot_urdf_path",
        type=Path,
        default=None,
        help=(
            "Robot URDF path. Defaults to the vendored FR3 URDF under fr3/ "
            "when --robot fr3. The default is the local FR3 gripper URDF."
        ),
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
    parser.add_argument("--link_x_lower", type=float, default=-0.2)
    parser.add_argument("--link_y_lower", type=float, default=-0.4)
    parser.add_argument("--link_y_upper", type=float, default=0.4)
    parser.add_argument("--link_z_lower", type=float, default=0.1)
    parser.add_argument("--disable_link_y_bounds", action="store_true")
    parser.add_argument(
        "--self_collision_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Check possible Franka self-collision using the saved collision "
            "spheres for the configured FR3 body pairs."
        ),
    )
    parser.add_argument(
        "--self_collision_stride",
        type=int,
        default=None,
        help="Sampling stride for self-collision checks. Defaults to camera_collision_stride.",
    )
    parser.add_argument(
        "--self_collision_clearance",
        type=float,
        default=0.0,
        help=(
            "Required minimum sphere clearance for the configured Franka "
            "self-collision pairs."
        ),
    )
    parser.add_argument(
        "--drake_xy_prism_height",
        type=float,
        default=None,
        help="Optional full z height for XY-prism camera obstacles. Defaults to physical camera-box height.",
    )
    parser.add_argument("--drake_physical_camera_height", action="store_true")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish the trajectory to Meshcat.",
    )
    parser.add_argument("--no_visualize", dest="visualize", action="store_false")
    parser.add_argument(
        "--start_button",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for a Meshcat button click before playing the trajectory.",
    )
    parser.add_argument("--no_start_button", dest="start_button", action="store_false")
    parser.add_argument("--start_button_name", type=str, default="Start playback")
    parser.add_argument(
        "--replay_button",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a Meshcat button that replays the trajectory after playback.",
    )
    parser.add_argument("--replay_button_name", type=str, default="Replay trajectory")
    parser.add_argument("--playback_stride", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--show_robot_samples",
        action="store_true",
        help="Draw the Drake robot sample specs used for workspace/collision checks.",
    )
    parser.add_argument(
        "--robot_sample_marker_radius",
        type=float,
        default=None,
        help=(
            "Radius of the Meshcat spheres for --show_robot_samples. "
            "Defaults to each saved collision sphere radius."
        ),
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of samples to generate when the trajectory argument is a Fourier folder.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="Duration to use when the trajectory argument is a Fourier folder.",
    )
    parser.add_argument(
        "--hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Meshcat alive after playback until Enter is pressed.",
    )
    parser.add_argument("--no_hold", dest="hold", action="store_false")
    parser.add_argument(
        "--stop_on_collision",
        action="store_true",
        help="Exit before playback if the sampled Drake collision margin is negative.",
    )
    args = parser.parse_args()

    traj_path = resolve_traj_path(args.trajectory)
    os.chdir(SCRIPT_DIR)

    t, q, dq, ddq = load_trajectory(
        traj_path,
        num_timesteps=args.num_timesteps,
        time_horizon=args.time_horizon,
    )
    logger.info(f"Loaded trajectory: {traj_path}")
    logger.info(f"Samples: {len(t)}, joints: {q.shape[1]}")

    xy_prism_height = (
        None if args.drake_physical_camera_height else args.drake_xy_prism_height
    )
    checker, collision_values = report_drake_collision(
        q,
        t,
        robot_name=args.robot,
        robot_urdf_path=args.robot_urdf_path,
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
        robot_urdf_path=args.robot_urdf_path,
        camera_chamfer_radius=args.drake_camera_chamfer_radius,
        xy_prism_height=xy_prism_height,
        camera_box_xy_scale=args.camera_box_xy_scale,
        camera_box_z_scale=args.camera_box_z_scale,
    )
    if q.shape[1] != plant.num_positions(model_instance):
        raise ValueError(
            f"Trajectory has {q.shape[1]} joints, but Drake model has "
            f"{plant.num_positions(model_instance)} positions"
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

    logger.info(
        f"Playing trajectory in Meshcat at speed={args.speed}, "
        f"playback_stride={args.playback_stride}"
    )

    def play_once():
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

    play_once()
    logger.info("Playback finished.")

    if args.hold:
        if args.replay_button:
            hold_with_replay_button(meshcat, args.replay_button_name, play_once)
        else:
            try:
                input("Press Enter to exit Meshcat playback...")
            except EOFError:
                logger.info("No stdin available; exiting Meshcat playback.")


if __name__ == "__main__":
    main()
