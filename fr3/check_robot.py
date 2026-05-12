import argparse
import tempfile
import time
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np


DEFAULT_URDF = (
    Path(__file__).resolve().parent
    / "robot_description"
    / "fr3_description"
    / "fr3_gripper.urdf"
)
DEFAULT_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
DEFAULT_INIT_Q = np.array([0.0, 0.0, 0.0, -1.52715, 0.0, 1.8675, 0.0])
DEFAULT_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the FR3 gripper URDF in MuJoCo at the initial joint pose."
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="FR3 URDF to load.",
    )
    parser.add_argument(
        "--init_q",
        type=float,
        nargs=7,
        default=DEFAULT_INIT_Q.tolist(),
        help="Initial FR3 arm joint positions.",
    )
    parser.add_argument(
        "--finger_width",
        type=float,
        default=0.02,
        help="Initial value for each gripper finger prismatic joint.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Only load and print the state; do not open the MuJoCo viewer.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance MuJoCo physics in the viewer. By default, hold the initial pose.",
    )
    return parser.parse_args()


def mujoco_urdf_path(urdf_path: Path) -> tuple[Path, Path | None]:
    """Return a MuJoCo-loadable URDF path and an optional temp file to delete."""
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)

    parse_error = None
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as exc:
        parse_error = exc
        tree = None

    if tree is None:
        lines = urdf_path.read_text(encoding="utf-8").splitlines()
        cleaned_lines = []
        removed_extra_link_close = False
        previous_nonempty = ""
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

        cleaned_xml = "\n".join(cleaned_lines) + "\n"
        tree = ET.ElementTree(ET.fromstring(cleaned_xml))

    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is not None:
                mesh = geometry.find("mesh")
                if mesh is not None:
                    mesh.attrib.pop("scale", None)

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".urdf",
        prefix=".mujoco_clean_",
        dir=urdf_path.parent,
        delete=False,
    )
    with temp_file:
        tree.write(temp_file, encoding="unicode", xml_declaration=True)
    return Path(temp_file.name), Path(temp_file.name)


def joint_qpos_address(model, mujoco, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint {joint_name!r} was not found in the MuJoCo model.")
    return int(model.jnt_qposadr[joint_id])


def set_joint_position(model, data, mujoco, joint_name: str, value: float) -> None:
    qpos_adr = joint_qpos_address(model, mujoco, joint_name)
    data.qpos[qpos_adr] = float(value)


def configure_initial_state(model, data, mujoco, init_q: np.ndarray, finger_width: float):
    model.opt.gravity[:] = 0.0
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    for joint_name, joint_position in zip(DEFAULT_ARM_JOINTS, init_q):
        set_joint_position(model, data, mujoco, joint_name, joint_position)

    for joint_name in DEFAULT_FINGER_JOINTS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0:
            set_joint_position(model, data, mujoco, joint_name, finger_width)

    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    init_q = np.asarray(args.init_q, dtype=float)

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `mujoco` Python package is not installed in this environment. "
            "Install it first, then rerun this script."
        ) from exc

    model_path, temp_path = mujoco_urdf_path(args.urdf)
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        configure_initial_state(
            model,
            data,
            mujoco,
            init_q=init_q,
            finger_width=args.finger_width,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"Loaded MuJoCo model: {args.urdf.expanduser().resolve()}")
    print(f"Gravity: {model.opt.gravity}")
    print(f"Initial arm q: {init_q}")
    print(f"qpos: {data.qpos.copy()}")

    if args.no_viewer:
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
