import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
    Sphere,
    SpatialInertia,
    UnitInertia,
)
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_DESCRIPTION_DIR = PROJECT_ROOT / "robot_description"

CAMERA_BOX_MARGIN_SCALE = 1.1
CAMERA_BOX_HEIGHT_SCALE = 1.1
CAMERA_BOX_SPECS_MM = (
    ("camera_1_pos_y", np.array([165.0, 340.0, 170.0]), np.array([170.0, 160.0, 340.0])),
    ("camera_2_pos_y", np.array([870.0, 365.0, 180.0]), np.array([180.0, 110.0, 360.0])),
    ("camera_1_neg_y", np.array([165.0, -340.0, 170.0]), np.array([170.0, 160.0, 340.0])),
    ("camera_2_neg_y", np.array([870.0, -365.0, 180.0]), np.array([180.0, 110.0, 360.0])),
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


def _strip_urdf_geometry(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        for tag in ("collision", "visual"):
            for elem in list(link.findall(tag)):
                link.remove(elem)

    for joint in root.findall("joint"):
        for elem in list(joint.findall("safety_controller")):
            joint.remove(elem)

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
    ):
        self.robot_name = robot_name
        self.robot_urdf_path = Path(robot_urdf_path) if robot_urdf_path else find_robot_urdf(robot_name)
        self.min_distance = float(min_distance)
        self.robot_sphere_radius = max(float(robot_sphere_radius), 1e-6)
        self.robot_link_samples = max(2, int(robot_link_samples))
        self.camera_chamfer_radius = max(float(camera_chamfer_radius), 0.0)
        self.camera_boxes = camera_boxes or default_camera_boxes(
            xy_prism_height=xy_prism_height,
            xy_scale=camera_box_xy_scale,
            z_scale=camera_box_z_scale,
        )

        self._build()

    def _build(self):
        urdf_root, urdf_xml = _strip_urdf_geometry(self.robot_urdf_path)
        root_link = _root_link_name(urdf_root)
        joint_segments = _joint_segments(urdf_root)

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        self.model_instance = Parser(self.plant).AddModelsFromString(urdf_xml, "urdf")[0]
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
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(self.root_context)
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
        sphere = Sphere(self.robot_sphere_radius)
        body_names = {
            self.plant.get_body(idx).name()
            for idx in self.plant.GetBodyIndices(self.model_instance)
        }

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

        for parent_name, _child_name, xyz in joint_segments:
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
                    f"{parent_name}_segment_capsule_collision",
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

        for body_name in sorted(body_names):
            body = self.plant.GetBodyByName(body_name, self.model_instance)
            specs.append((body, np.zeros(3)))

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
                specs.append((parent_body, alpha * xyz))
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
            (Box(max(size[0] - 2.0 * chamfer, 1e-6), size[1], size[2]), np.zeros(3), "x_strip"),
            (Box(size[0], max(size[1] - 2.0 * chamfer, 1e-6), size[2]), np.zeros(3), "y_strip"),
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
            points = self._robot_sample_points(q_i)
            min_y = float(np.min(points[:, 1]))
            max_y = float(np.max(points[:, 1]))
            margins.append(
                (
                    min_y - self.robot_sphere_radius - float(lower),
                    float(upper) - max_y - self.robot_sphere_radius,
                )
            )
        return np.asarray(margins, dtype=float)

    def robot_link_wall_margins(self, q, y_lower=-0.45, y_upper=0.35, z_lower=0.0):
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        margins = []
        for q_i in q:
            points = self._robot_sample_points(q_i)
            min_y = float(np.min(points[:, 1]))
            max_y = float(np.max(points[:, 1]))
            min_z = float(np.min(points[:, 2]))
            margins.append(
                (
                    min_y - self.robot_sphere_radius - float(y_lower),
                    float(y_upper) - max_y - self.robot_sphere_radius,
                    min_z - float(z_lower),
                )
            )
        return np.asarray(margins, dtype=float)

    def _robot_sample_points(self, q):
        self.plant.SetPositions(self.plant_context, self.model_instance, q)
        points = []
        for body, offset in self._robot_sample_specs:
            x_wb = self.plant.EvalBodyPoseInWorld(self.plant_context, body)
            points.append(x_wb.multiply(offset))
        return np.asarray(points, dtype=float)

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
            pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(float(max_distance))
            if not pairs:
                min_distance = min(min_distance, float(max_distance))
            else:
                min_distance = min(min_distance, min(pair.distance for pair in pairs))
        return float(min_distance)
