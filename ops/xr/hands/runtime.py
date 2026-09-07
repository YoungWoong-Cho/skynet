"""Install one Skynet URDF hand into the pinned DexVerse process, after AppLauncher."""

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from types import ModuleType


def read_bundle(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "skynet.simulation-hand/v1":
        raise ValueError("Unsupported simulation hand bundle")
    for name, expected in manifest["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError("Simulation hand asset is missing: " + name)
        if (
            p.stat().st_size != expected["size_bytes"]
            or hashlib.sha256(p.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Simulation hand asset checksum mismatch: " + name)
    return root, manifest


def bounded_fingers(values, limits):
    import numpy as np

    array = np.asarray(values)
    bounds = np.asarray(limits)
    if array.shape != (len(bounds),) or not np.isfinite(array).all():
        raise ValueError("Hand retargeting returned missing or nonfinite joint values")
    return np.clip(array, bounds[:, 0], bounds[:, 1])


def adjacent_collision_pairs(urdf, depth=2):
    """Exclude overlapping mounts and fixed sensors along the same finger chain."""
    xml = ET.parse(urdf).getroot()
    parents = {
        j.find("child").get("link"): (
            j.find("parent").get("link"),
            j.get("type") != "fixed",
        )
        for j in xml.findall("joint")
    }
    pairs = []
    for child in parents:
        node, joints = child, 0
        while node in parents:
            parent, moving = parents[node]
            joints += int(moving)
            if joints > depth:
                break
            pairs.append((child, parent))
            node = parent
    return pairs


def filter_adjacent_collisions(urdf, usd, depth=2):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd)
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise ValueError(
            "Hand URDF conversion failed; inspect the converter error above"
        )
    bodies = {
        p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for child, parent in adjacent_collision_pairs(urdf, depth):
        if child in bodies and parent in bodies:
            # https://openusd.org/release/wp_rigid_body_physics.html#pair-filtering
            UsdPhysics.FilteredPairsAPI.Apply(
                bodies[child]
            ).CreateFilteredPairsRel().AddTarget(bodies[parent].GetPath())
    stage.GetRootLayer().Save()


def install(directory):
    root, m = read_bundle(directory)
    import isaaclab.sim as sim
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
    from isaaclab.utils import configclass
    from dexverse.robot_agents import TabletopRobotSetup
    from dexverse.tasks import dexverse_base_env_cfg as base
    from dexverse.devices.retargeters import simple_relative_retargeting as retargeting

    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(root / "simulation.urdf"),
            usd_dir=str(root / "usd"),
            usd_file_name="hand.usd",
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            make_instanceable=False,
            self_collision=True,
            collision_from_visuals=False,
            collider_type="convex_hull",
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=10.0, damping=0.2
                ),
            ),
        )
    )
    filter_adjacent_collisions(
        root / "simulation.urdf", converter.usd_path, m["collision_neighbor_depth"]
    )
    wrist, fingers = m["wrist_joints"], m["finger_joints"]

    @configclass
    class Actions:
        translation = JointPositionActionCfg(
            asset_name="robot",
            joint_names=wrist[:3],
            preserve_order=True,
            use_default_offset=True,
        )
        rotation = JointPositionActionCfg(
            asset_name="robot",
            joint_names=wrist[3:],
            preserve_order=True,
            use_default_offset=True,
        )
        fingers_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=fingers,
            preserve_order=True,
            use_default_offset=False,
        )

    articulation = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(
            usd_path=converter.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1.0
            ),
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.75, 0.0, 0.5),
            joint_pos=dict(
                m["neutral"],
                skynet_x=0.5,
                skynet_y=0.0,
                skynet_z=0.3,
                skynet_yaw=0.0,
                skynet_pitch=0.0,
                skynet_roll=0.0,
            ),
        ),
        actuators={
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=wrist,
                stiffness=2000.0,
                damping=400.0,
                armature=0.01,
                effort_limit_sim=30.0,
                velocity_limit_sim=5.0,
            ),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=fingers,
                stiffness=10.0,
                damping=0.2,
                armature=0.01,
                effort_limit_sim=2.0,
                velocity_limit_sim=5.0,
            ),
        },
    )

    def builder():
        return TabletopRobotSetup(
            robot_config_kwargs=dict(
                palm_body_name=m["palm"],
                fingertip_body_names=m["tips"],
                hand_tips_body_names=[m["palm"], *m["tips"]],
                wrist_joint_name="skynet_(yaw|pitch|roll)",
                arm_joint_names_expr=["skynet_(x|y|z)"],
                setup_contact_sensors=True,
            ),
            scene_robot=articulation,
            actions=Actions(),
            controller_mode="joint",
            teleop_config=dict(
                hand_joint_names=fingers,
                wrist_position_offset=(-0.25, 0.0, 0.8),
                retargeter_config_filename=m["robot"],
                retargeter_urdf_path=None,
                apply_shadow_specific_postprocess=False,
            ),
        )

    # A headset's wrist axes need not follow the middle finger. Derive the
    # imported hand's finger frame from the knuckles to avoid a shared yaw bias.
    original_canonical = (
        retargeting.SimpleRelativeRetargeter._convert_hand_to_canonical_joint_positions
    )

    def canonical(self, hand_data, hand=None):
        if self.cfg.robot_type != m["robot"]:
            return original_canonical(self, hand_data, hand)
        import numpy as np
        from anatomy import canonical_points

        names = retargeting.DEX_RETARGETING_HAND_JOINT_NAMES
        if not isinstance(hand_data, dict) or any(n not in hand_data for n in names):
            return None
        points = np.asarray([hand_data[n][:3] for n in names])
        try:
            return canonical_points(points, m["side"])
        except ValueError:
            return None  # The collection alignment gate prevents invalid tracking from recording.

    retargeting.SimpleRelativeRetargeter._convert_hand_to_canonical_joint_positions = (
        canonical
    )

    original_reference = retargeting.SimpleRelativeRetargeter._compute_dex_ref_value

    def reference(self, solver, points):
        if (
            self.cfg.robot_type != m["robot"]
            or m.get("retargeting_mode") != "finger_segments"
        ):
            return original_reference(self, solver, points)
        import numpy as np
        from anatomy import segment_targets

        if not hasattr(solver, "_skynet_segment_lengths"):
            optimizer = solver.optimizer
            robot = optimizer.robot
            neutral = np.array(
                [m["neutral"].get(name, 0.0) for name in robot.dof_joint_names]
            )
            robot.compute_forward_kinematics(neutral)
            solver._skynet_segment_lengths = np.array(
                [
                    np.linalg.norm(
                        robot.get_link_pose(robot.get_link_index(end))[:3, 3]
                        - robot.get_link_pose(robot.get_link_index(start))[:3, 3]
                    )
                    for start, end in zip(
                        optimizer.origin_link_names, optimizer.task_link_names
                    )
                ]
            )
        return segment_targets(
            points,
            solver.optimizer.target_link_human_indices,
            solver._skynet_segment_lengths,
        )

    retargeting.SimpleRelativeRetargeter._compute_dex_ref_value = reference

    original_assign = retargeting.SimpleRelativeRetargeter._assign_hand_fingers

    def assign(self, action, hand, finger_values):
        if self.cfg.robot_type == m["robot"]:
            mapping = self._dex_to_action_finger_indices.get(hand)
            if (
                mapping is None
                or len(mapping) != len(fingers)
                or any(
                    index is None or index < 0 or index >= len(finger_values)
                    for index in mapping
                )
            ):
                raise ValueError("Imported hand has an incomplete finger joint mapping")
        original_assign(self, action, hand, finger_values)
        if self.cfg.robot_type == m["robot"]:
            indices = self._layout["hands"][hand]["finger_indices"]
            action[list(indices)] = bounded_fingers(
                action[list(indices)], [m["finger_limits"][name] for name in fingers]
            )

    retargeting.SimpleRelativeRetargeter._assign_hand_fingers = assign

    original = base._get_tabletop_robot_setup_builders
    base._get_tabletop_robot_setup_builders = lambda: dict(
        original(), **{m["robot"]: builder}
    )
    module_name = "skynet_imported_" + m["robot"]
    module = ModuleType(module_name)
    module.LAYOUT = {
        "output_dim": m["action_dimension"],
        "hands": {
            m["side"]: {
                "wrist_trans_indices": (0, 1, 2),
                "wrist_rot_indices": (3, 4, 5),
                "wrist_rot_order": "xyz",
                "wrist_rot_signs": (1.0, 1.0, 1.0),
                "finger_indices": tuple(range(6, m["action_dimension"])),
                "finger_joint_names": fingers,
                "finger_permutation": tuple(range(len(fingers))),
            }
        },
    }
    module.SIMPLE_RELATIVE_DEX_RETARGETING = {
        "hands": {
            m["side"]: {
                "config_paths": {m["retargeting_scheme"]: str(root / "retarget.json")},
                "urdf_path": str(root / "retarget.urdf"),
            }
        }
    }
    sys.modules[module_name] = module
    retargeting.SIMPLE_RETARGETER_LAYOUT_SOURCES[m["robot"]] = (module_name, "LAYOUT")
    return m


def validate_environment(env, manifest):
    import numpy as np

    if env is None:
        raise ValueError(
            "DexVerse did not create an environment for the selected hand; inspect the preceding error"
        )
    robot = env.unwrapped.scene["robot"]
    expected = [*manifest["wrist_joints"], *manifest["finger_joints"]]
    missing = set(expected) - set(robot.joint_names)
    missing_bodies = set([manifest["palm"], *manifest["tips"]]) - set(robot.body_names)
    if missing or missing_bodies:
        raise ValueError(
            f"Imported hand is missing joints {sorted(missing)} or bodies {sorted(missing_bodies)}"
        )
    if env.unwrapped.action_manager.total_action_dim != len(expected):
        raise ValueError(
            "Imported hand action dimension differs from its retargeting map"
        )
    if not np.isfinite(robot.data.joint_pos.detach().cpu().numpy()).all():
        raise ValueError("Imported hand produced invalid initial joint positions")
