import os
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from loguru import logger
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Capsule,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    MinimumDistanceLowerBoundConstraint,
    Parser,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
    Sphere,
    UnitInertia,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DESCRIPTION_DIR = PROJECT_ROOT / "robot_description"
DEFAULT_COLLISION_SPHERES_PATH = (
    ROBOT_DESCRIPTION_DIR / "fr3_description" / "fr3_collision_spheres.py"
)

CAMERA_BOX_MARGIN_SCALE = 1.1
CAMERA_BOX_HEIGHT_SCALE = 1.1
CAMERA_BOX_SPECS_MM = (
    (
        "camera_1_pos_y",
        np.array([165.0, 340.0, 170.0]),
        np.array([170.0, 160.0, 340.0]),
    ),
    (
        "camera_2_pos_y",
        np.array([870.0, 365.0, 180.0]),
        np.array([180.0, 110.0, 360.0]),
    ),
    (
        "camera_1_neg_y",
        np.array([165.0, -340.0, 170.0]),
        np.array([170.0, 160.0, 340.0]),
    ),
    (
        "camera_2_neg_y",
        np.array([870.0, -365.0, 180.0]),
        np.array([180.0, 110.0, 360.0]),
    ),
)
DEFAULT_FIXED_COLLISION_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
COLLISION_SPHERE_BODY_NAME_ALIASES = {
    "panda_finger_soft_l": "panda_leftfinger",
    "panda_finger_soft_r": "panda_rightfinger",
}
ROBOT_BODY_NAME_ALIASES = {
    **COLLISION_SPHERE_BODY_NAME_ALIASES,
    **{f"fr3_link{idx}": f"panda_link{idx}" for idx in range(8)},
    "fr3_hand": "panda_hand",
    "fr3_leftfinger": "panda_leftfinger",
    "fr3_rightfinger": "panda_rightfinger",
}
DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS = (
    ("fr3_link0", "fr3_link5"),
    ("fr3_link0", "fr3_link6"),
    ("fr3_link0", "fr3_link7"),
    ("fr3_link0", "fr3_hand"),
    ("fr3_link0", "fr3_leftfinger"),
    ("fr3_link0", "fr3_rightfinger"),
    ("fr3_link1", "fr3_link5"),
    ("fr3_link1", "fr3_link6"),
    ("fr3_link1", "fr3_link7"),
    ("fr3_link1", "fr3_hand"),
    ("fr3_link1", "fr3_leftfinger"),
    ("fr3_link1", "fr3_rightfinger"),
    ("fr3_link2", "fr3_link5"),
    ("fr3_link2", "fr3_link7"),
    ("fr3_link2", "fr3_hand"),
    ("fr3_link2", "fr3_leftfinger"),
    ("fr3_link2", "fr3_rightfinger"),
    ("fr3_link5", "fr3_hand"),
    ("fr3_link5", "fr3_leftfinger"),
    ("fr3_link5", "fr3_rightfinger"),
)


@dataclass(frozen=True)
class CameraBox:
    name: str
    center: np.ndarray
    size: np.ndarray


def find_robot_urdf(robot_name):
    urdf_name = f"{robot_name}.urdf"
    for root, _dirs, files in os.walk(ROBOT_DESCRIPTION_DIR):
        if urdf_name in files:
            return Path(root) / urdf_name
    raise FileNotFoundError(f"Could not find {urdf_name} under {ROBOT_DESCRIPTION_DIR}")


def default_camera_boxes(
    xy_prism_height=None,
    xy_scale=CAMERA_BOX_MARGIN_SCALE,
    z_scale=CAMERA_BOX_HEIGHT_SCALE,
):
    boxes = []
    for name, center_mm, size_mm in CAMERA_BOX_SPECS_MM:
        center = center_mm / 1000.0
        size = size_mm.astype(float) / 1000.0
        size[:2] *= float(xy_scale)
        if xy_prism_height is not None:
            size[2] = float(xy_prism_height) * float(z_scale)
        else:
            size[2] *= float(z_scale)
        boxes.append(CameraBox(name=name, center=center, size=size))
    return boxes


def _parse_urdf_tolerant(urdf_path):
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


def _fix_collision_only_joints(urdf_root, joint_names=DEFAULT_FIXED_COLLISION_JOINTS):
    joint_names = set(joint_names)
    for joint in urdf_root.findall("joint"):
        if joint.attrib.get("name") not in joint_names:
            continue
        joint.attrib["type"] = "fixed"
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for elem in list(joint.findall(tag)):
                joint.remove(elem)


def _strip_urdf_geometry(urdf_path):
    tree = _parse_urdf_tolerant(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        for tag in ("collision", "visual"):
            for elem in list(link.findall(tag)):
                link.remove(elem)

    for joint in root.findall("joint"):
        for elem in list(joint.findall("safety_controller")):
            joint.remove(elem)

    _fix_collision_only_joints(root)

    return root, ET.tostring(root, encoding="unicode")


def _root_link_name(urdf_root):
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


def _joint_segments(urdf_root):
    segments = []
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = np.zeros(3)
        if origin is not None and "xyz" in origin.attrib:
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=float)
        segments.append((parent.attrib["link"], child.attrib["link"], xyz))
    return segments


def _geometry_set(geometry_ids):
    geometry_set = GeometrySet()
    geometry_set.Add(list(geometry_ids))
    return geometry_set


def normalize_robot_body_name(body_name):
    return ROBOT_BODY_NAME_ALIASES.get(body_name, body_name)


def normalize_robot_body_pairs(body_pairs):
    return tuple(
        (normalize_robot_body_name(first), normalize_robot_body_name(second))
        for first, second in body_pairs
    )


def _flatten_radius(radius_entry):
    radius = np.asarray(radius_entry, dtype=float).reshape(-1)
    if radius.size != 1:
        raise ValueError(f"Expected one radius value, got {radius_entry}")
    return float(radius[0])


def load_robot_collision_spheres(path=DEFAULT_COLLISION_SPHERES_PATH):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {}

    spec = spec_from_file_location("fr3_collision_spheres", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import collision sphere specs from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    spheres = {}
    for attr_name, positions in vars(module).items():
        if not attr_name.endswith("_positions"):
            continue
        sphere_name = attr_name[: -len("_positions")]
        radius_name = f"{sphere_name}_radius"
        if not hasattr(module, radius_name):
            raise ValueError(f"Missing {radius_name} in {path}")
        body_name = COLLISION_SPHERE_BODY_NAME_ALIASES.get(sphere_name, sphere_name)
        radii = getattr(module, radius_name)
        if len(positions) != len(radii):
            raise ValueError(
                f"{sphere_name} has {len(positions)} positions but "
                f"{len(radii)} radii."
            )
        spheres[body_name] = [
            (np.asarray(position, dtype=float), _flatten_radius(radius))
            for position, radius in zip(positions, radii)
        ]
    return spheres


class DrakeCameraCollisionChecker:
    def __init__(
        self,
        robot_name="fr3",
        robot_urdf_path=None,
        min_distance=0.0,
        robot_sphere_radius=0.05,
        robot_link_samples=7,
        camera_boxes=None,
        camera_chamfer_radius=0.02,
        xy_prism_height=None,
        camera_box_xy_scale=CAMERA_BOX_MARGIN_SCALE,
        camera_box_z_scale=CAMERA_BOX_HEIGHT_SCALE,
        robot_collision_sphere_path=DEFAULT_COLLISION_SPHERES_PATH,
        use_saved_collision_spheres=True,
    ):
        self.robot_name = robot_name
        self.robot_urdf_path = (
            Path(robot_urdf_path) if robot_urdf_path else find_robot_urdf(robot_name)
        )
        self.min_distance = float(min_distance)
        self.robot_sphere_radius = max(float(robot_sphere_radius), 1e-6)
        self.robot_link_samples = max(2, int(robot_link_samples))
        self.camera_chamfer_radius = max(float(camera_chamfer_radius), 0.0)
        self.camera_boxes = camera_boxes or default_camera_boxes(
            xy_prism_height=xy_prism_height,
            xy_scale=camera_box_xy_scale,
            z_scale=camera_box_z_scale,
        )
        self.robot_collision_spheres = (
            load_robot_collision_spheres(robot_collision_sphere_path)
            if use_saved_collision_spheres
            else {}
        )

        self._build()

    def _build(self):
        urdf_root, urdf_xml = _strip_urdf_geometry(self.robot_urdf_path)
        root_link = _root_link_name(urdf_root)
        joint_segments = _joint_segments(urdf_root)

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        self.model_instance = Parser(self.plant).AddModelsFromString(urdf_xml, "urdf")[
            0
        ]
        self.camera_model_instance = self.plant.AddModelInstance("camera_boxes")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(root_link, self.model_instance).body_frame(),
        )

        friction = CoulombFriction(1.0, 1.0)
        robot_geometry_ids = self._register_robot_spheres(joint_segments, friction)
        camera_geometry_ids = self._register_camera_boxes(friction)
        self._robot_sample_specs = self._robot_link_sample_specs(joint_segments)

        self.scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration()
            .ExcludeWithin(_geometry_set(robot_geometry_ids))
            .ExcludeWithin(_geometry_set(camera_geometry_ids))
        )

        self.plant.Finalize()
        self.diagram = builder.Build()
        self.root_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.root_context)
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(
            self.root_context
        )
        self.distance_constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=self.min_distance,
            plant_context=self.plant_context,
        )
        logger.info(
            "Built Drake collision checker with "
            f"{len(robot_geometry_ids)} robot collision geometries and "
            f"{len(camera_geometry_ids)} camera collision geometries."
        )

    def _register_robot_spheres(self, joint_segments, friction):
        ids = []
        body_names = {
            self.plant.get_body(idx).name()
            for idx in self.plant.GetBodyIndices(self.model_instance)
        }
        if self.robot_collision_spheres:
            missing_body_names = sorted(set(self.robot_collision_spheres) - body_names)
            if missing_body_names:
                logger.warning(
                    "Skipping collision sphere specs for missing bodies: %s",
                    missing_body_names,
                )
            for body_name in sorted(set(self.robot_collision_spheres) & body_names):
                body = self.plant.GetBodyByName(body_name, self.model_instance)
                for idx, (offset, radius) in enumerate(
                    self.robot_collision_spheres[body_name]
                ):
                    ids.append(
                        self.plant.RegisterCollisionGeometry(
                            body,
                            RigidTransform(offset),
                            Sphere(radius),
                            f"{body_name}_saved_sphere_{idx}_collision",
                            friction,
                        )
                    )
            return ids

        sphere = Sphere(self.robot_sphere_radius)
        for body_name in sorted(body_names):
            body = self.plant.GetBodyByName(body_name, self.model_instance)
            ids.append(
                self.plant.RegisterCollisionGeometry(
                    body,
                    RigidTransform(),
                    sphere,
                    f"{body_name}_point_collision",
                    friction,
                )
            )

        for parent_name, child_name, xyz in joint_segments:
            if parent_name == "world" or parent_name not in body_names:
                continue
            segment_length = np.linalg.norm(xyz)
            if segment_length <= 1e-12:
                continue
            parent_body = self.plant.GetBodyByName(parent_name, self.model_instance)
            ids.append(
                self.plant.RegisterCollisionGeometry(
                    parent_body,
                    RigidTransform(
                        RotationMatrix.MakeFromOneVector(xyz, 2),
                        0.5 * xyz,
                    ),
                    Capsule(self.robot_sphere_radius, segment_length),
                    f"{parent_name}_to_{child_name}_segment_capsule_collision",
                    friction,
                )
            )
        return ids

    def _robot_link_sample_specs(self, joint_segments):
        specs = []
        body_names = {
            self.plant.get_body(idx).name()
            for idx in self.plant.GetBodyIndices(self.model_instance)
        }
        if self.robot_collision_spheres:
            for body_name in sorted(set(self.robot_collision_spheres) & body_names):
                body = self.plant.GetBodyByName(body_name, self.model_instance)
                for offset, radius in self.robot_collision_spheres[body_name]:
                    specs.append((body, offset, radius))
            return specs

        for body_name in sorted(body_names):
            body = self.plant.GetBodyByName(body_name, self.model_instance)
            specs.append((body, np.zeros(3), self.robot_sphere_radius))

        for parent_name, _child_name, xyz in joint_segments:
            if parent_name == "world" or parent_name not in body_names:
                continue
            if np.linalg.norm(xyz) <= 1e-12:
                continue
            parent_body = self.plant.GetBodyByName(parent_name, self.model_instance)
            for alpha in np.linspace(
                0.0,
                1.0,
                self.robot_link_samples,
                endpoint=True,
            )[1:]:
                specs.append((parent_body, alpha * xyz, self.robot_sphere_radius))
        return specs

    def _register_camera_boxes(self, friction):
        ids = []
        inertia = SpatialInertia(
            mass=1.0,
            p_PScm_E=np.zeros(3),
            G_SP_E=UnitInertia.SolidBox(1.0, 1.0, 1.0),
        )
        for box in self.camera_boxes:
            body = self.plant.AddRigidBody(
                box.name,
                self.camera_model_instance,
                inertia,
            )
            self.plant.WeldFrames(
                self.plant.world_frame(),
                body.body_frame(),
                RigidTransform(box.center),
            )
            ids.extend(self._register_camera_geometry(body, box, friction))
        return ids

    def _register_camera_geometry(self, body, box, friction):
        size = np.asarray(box.size, dtype=float)
        chamfer = min(self.camera_chamfer_radius, 0.49 * min(size[0], size[1]))
        if chamfer <= 0.0:
            return [
                self.plant.RegisterCollisionGeometry(
                    body,
                    RigidTransform(),
                    Box(*size),
                    f"{box.name}_box_collision",
                    friction,
                )
            ]

        ids = []
        strip_shapes = (
            (
                Box(max(size[0] - 2.0 * chamfer, 1e-6), size[1], size[2]),
                np.zeros(3),
                "x_strip",
            ),
            (
                Box(size[0], max(size[1] - 2.0 * chamfer, 1e-6), size[2]),
                np.zeros(3),
                "y_strip",
            ),
        )
        for shape, offset, suffix in strip_shapes:
            ids.append(
                self.plant.RegisterCollisionGeometry(
                    body,
                    RigidTransform(offset),
                    shape,
                    f"{box.name}_{suffix}_collision",
                    friction,
                )
            )

        corner_offsets = (
            np.array([size[0] / 2.0 - chamfer, size[1] / 2.0 - chamfer, 0.0]),
            np.array([size[0] / 2.0 - chamfer, -size[1] / 2.0 + chamfer, 0.0]),
            np.array([-size[0] / 2.0 + chamfer, size[1] / 2.0 - chamfer, 0.0]),
            np.array([-size[0] / 2.0 + chamfer, -size[1] / 2.0 + chamfer, 0.0]),
        )
        corner_shape = Cylinder(chamfer, size[2])
        for idx, offset in enumerate(corner_offsets):
            ids.append(
                self.plant.RegisterCollisionGeometry(
                    body,
                    RigidTransform(offset),
                    corner_shape,
                    f"{box.name}_corner_{idx}_collision",
                    friction,
                )
            )
        return ids

    def constraint_values(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return np.array(
            [self._signed_distance_margin(q_i) - self.min_distance for q_i in q],
            dtype=float,
        )

    def minimum_distance_constraint_values(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        return np.array(
            [float(self.distance_constraint.Eval(q_i)[0]) for q_i in q],
            dtype=float,
        )

    def _signed_distance_margin(self, q):
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        query_object = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph_context
        )
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(10.0)
        if not pairs:
            return 10.0
        return float(min(pair.distance for pair in pairs))

    def robot_link_y_margins(self, q, lower=-0.35, upper=0.35):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        margins = []
        for q_i in q:
            points, radii = self._robot_sample_points_and_radii(q_i)
            margins.append(
                (
                    float(np.min(points[:, 1] - radii)) - float(lower),
                    float(upper) - float(np.max(points[:, 1] + radii)),
                )
            )
        return np.asarray(margins, dtype=float)

    def robot_link_wall_margins(self, q, y_lower=-0.45, y_upper=0.35, z_lower=0.0):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        margins = []
        for q_i in q:
            points, radii = self._robot_sample_points_and_radii(q_i)
            margins.append(
                (
                    float(np.min(points[:, 1] - radii)) - float(y_lower),
                    float(y_upper) - float(np.max(points[:, 1] + radii)),
                    float(np.min(points[:, 2] - radii)) - float(z_lower),
                )
            )
        return np.asarray(margins, dtype=float)

    def _robot_sample_points(self, q):
        points, _radii = self._robot_sample_points_and_radii(q)
        return points

    def _robot_sample_points_and_radii(self, q):
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        points = []
        radii = []
        for body, offset, radius in self._robot_sample_specs:
            x_wb = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            points.append(x_wb.multiply(offset))
            radii.append(float(radius))
        return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)

    def _robot_sample_points_by_body(self, q):
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        samples_by_body = {}
        for body, offset, radius in self._robot_sample_specs:
            x_wb = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            samples_by_body.setdefault(body.name(), []).append(
                (x_wb.multiply(offset), float(radius))
            )
        return samples_by_body

    def robot_self_collision_pair_margins(
        self,
        q,
        body_pairs=DEFAULT_FR3_SELF_COLLISION_BODY_PAIRS,
    ):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        body_pairs = normalize_robot_body_pairs(body_pairs)
        margins = []
        for q_i in q:
            samples_by_body = self._robot_sample_points_by_body(q_i)
            q_margins = []
            for first_body, second_body in body_pairs:
                first_samples = samples_by_body.get(first_body)
                second_samples = samples_by_body.get(second_body)
                if first_samples is None or second_samples is None:
                    missing = [
                        body
                        for body, samples in (
                            (first_body, first_samples),
                            (second_body, second_samples),
                        )
                        if samples is None
                    ]
                    raise ValueError(
                        "Missing self-collision body samples for "
                        f"{missing}. Check robot_urdf_path and collision spheres."
                    )

                pair_margin = np.inf
                for first_point, first_radius in first_samples:
                    for second_point, second_radius in second_samples:
                        margin = (
                            float(np.linalg.norm(first_point - second_point))
                            - float(first_radius)
                            - float(second_radius)
                        )
                        pair_margin = min(pair_margin, margin)
                q_margins.append(pair_margin)
            margins.append(q_margins)
        return np.asarray(margins, dtype=float)

    def min_clearance(self, q, max_distance=10.0):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        min_distance = np.inf
        for q_i in q:
            self.plant.SetPositions(self.plant_context, self.model_instance, q_i)
            query_object = self.scene_graph.get_query_output_port().Eval(
                self.scene_graph_context
            )
            pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(
                float(max_distance)
            )
            if not pairs:
                min_distance = min(min_distance, float(max_distance))
            else:
                min_distance = min(min_distance, min(pair.distance for pair in pairs))
        return float(min_distance)
