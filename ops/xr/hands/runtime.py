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
    hand_layouts(manifest)
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


def configure_mimic_constraints(usd, joints):
    """Keep URDF mechanical couplings exact in the converted PhysX model."""
    if not joints:
        return
    import math
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd)
    prims = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    for joint in joints:
        prim = prims.get(joint["name"])
        reference = prims.get(joint["mimic"]["joint"])
        if not prim or not reference:
            raise ValueError("Converted hand is missing a coupled finger joint")
        prefix = "physxMimicJoint:rot" + prim.GetAttribute("physics:axis").Get()
        gearing = prim.GetAttribute(prefix + ":gearing")
        targets = prim.GetRelationship(prefix + ":referenceJoint").GetTargets()
        if not gearing or targets != [reference.GetPath()]:
            raise ValueError("Simulator did not import the hand's mimic relationship")
        # PhysX: q + gearing*q_reference + offset = 0, in degrees.
        # URDF: q = multiplier*q_reference + offset, in radians.
        gearing.Set(-float(joint["mimic"].get("multiplier", 1)))
        prim.GetAttribute(prefix + ":offset").Set(
            -math.degrees(float(joint["mimic"].get("offset", 0)))
        )
        # The importer adds a weak, underdamped spring; the source specifies a
        # mechanical linkage. Zero frequency selects the rigid constraint.
        prim.GetAttribute(prefix + ":naturalFrequency").Set(0.0)
        prim.GetAttribute(prefix + ":dampingRatio").Set(0.0)
        prim.GetAttribute("physics:lowerLimit").Set(math.degrees(joint["lower"]))
        prim.GetAttribute("physics:upperLimit").Set(math.degrees(joint["upper"]))
    stage.GetRootLayer().Save()


def hand_layouts(manifest):
    """One common six-axis contract; retain support for archived v1 bundles."""
    layouts = manifest.get("hands") or {
        manifest["side"]: {
            "palm": manifest["palm"],
            "tips": manifest["tips"],
            "control_frame": "skynet_palm",
            "wrist_joints": manifest["wrist_joints"],
            "finger_joints": manifest["finger_joints"],
        }
    }
    order = manifest.get("hand_order") or sorted(
        layouts,
        key=lambda side: manifest["wrist_joints"].index(
            layouts[side]["wrist_joints"][0]
        ),
    )
    if len(order) != len(set(order)) or set(order) != set(layouts):
        raise ValueError("Hand order does not match the hand layout")
    layouts = {side: layouts[side] for side in order}
    if set(layouts) not in ({"right"}, {"left"}, {"right", "left"}):
        raise ValueError("Unsupported hand side layout")
    for layout in layouts.values():
        if len(layout["wrist_joints"]) != 6 or not layout["finger_joints"]:
            raise ValueError("Each hand requires six wrist axes and finger joints")
    if [n for h in layouts.values() for n in h["wrist_joints"]] != manifest[
        "wrist_joints"
    ] or [n for h in layouts.values() for n in h["finger_joints"]] != manifest[
        "finger_joints"
    ]:
        raise ValueError("Hand layout order differs from the action contract")
    names = manifest["wrist_joints"] + manifest["finger_joints"]
    if len(set(names)) != len(names) or len(names) != manifest["action_dimension"]:
        raise ValueError("Invalid hand action contract")
    return layouts


def initial_joint_positions(manifest):
    positions = dict(manifest["neutral"])
    layouts = hand_layouts(manifest)
    for side, layout in layouts.items():
        y = (-0.3 if side == "right" else 0.3) if len(layouts) == 2 else 0.0
        positions.update(zip(layout["wrist_joints"], (0.5, y, 0.3, 0.0, 0.0, 0.0)))
    return positions


def action_terms(manifest, action_type):
    # Field order is the stored action order: all wrists, then all fingers.
    terms = {
        side + "_wrist": action_type(
            asset_name="robot",
            joint_names=h["wrist_joints"],
            preserve_order=True,
            use_default_offset=True,
        )
        for side, h in hand_layouts(manifest).items()
    }
    terms["fingers"] = action_type(
        asset_name="robot",
        joint_names=manifest["finger_joints"],
        preserve_order=True,
        use_default_offset=False,
    )
    return terms


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
            # This Isaac Lab version forwards the misleadingly named option to
            # URDF parse_mimic. True is required to create PhysX mimic constraints.
            convert_mimic_joints_to_normal_joints=bool(m["mimic_joints"]),
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
    configure_mimic_constraints(converter.usd_path, m["mimic_joints"])
    filter_adjacent_collisions(
        root / "simulation.urdf", converter.usd_path, m["collision_neighbor_depth"]
    )
    wrist, fingers = m["wrist_joints"], m["finger_joints"]
    layouts = hand_layouts(m)

    terms = action_terms(m, JointPositionActionCfg)
    Actions = configclass(
        type(
            "Actions",
            (),
            {
                "__annotations__": {key: JointPositionActionCfg for key in terms},
                **terms,
            },
        )
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
            joint_pos=initial_joint_positions(m),
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
                right_palm_body_name=layouts.get("right", {}).get("palm"),
                left_palm_body_name=layouts["left"]["palm"]
                if len(layouts) == 2
                else None,
                wrist_joint_name="("
                + "|".join(n for h in layouts.values() for n in h["wrist_joints"][3:])
                + ")",
                arm_joint_names_expr=[
                    n for h in layouts.values() for n in h["wrist_joints"][:3]
                ],
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

    original = base._get_tabletop_robot_setup_builders
    base._get_tabletop_robot_setup_builders = lambda: dict(
        original(), **{m["robot"]: builder}
    )
    module_name = "skynet_imported_" + m["robot"]
    module = ModuleType(module_name)
    action_names = wrist + fingers
    module.LAYOUT = {
        "output_dim": m["action_dimension"],
        "hands": {
            side: {
                "wrist_trans_indices": tuple(
                    action_names.index(n) for n in h["wrist_joints"][:3]
                ),
                "wrist_rot_indices": tuple(
                    action_names.index(n) for n in h["wrist_joints"][3:]
                ),
                "wrist_rot_order": "xyz",
                "wrist_rot_signs": (1.0, 1.0, 1.0),
                "finger_indices": tuple(
                    action_names.index(n) for n in h["finger_joints"]
                ),
                "finger_joint_names": h["finger_joints"],
                "finger_permutation": tuple(range(len(h["finger_joints"]))),
            }
            for side, h in layouts.items()
        },
    }
    from dexverse.devices.wrist_origin import compute_wrist_joint_origin

    module.SIMPLE_ABSOLUTE_WRIST_ORIGIN = {
        side: compute_wrist_joint_origin(
            articulation, h["wrist_joints"][:3], h["wrist_joints"][3:]
        )
        for side, h in layouts.items()
    }
    module.SIMPLE_RELATIVE_DEX_RETARGETING = {
        "hands": {
            side: {
                "config_paths": {
                    m["retargeting_scheme"]: str(
                        root
                        / (
                            "retarget-" + side + ".json"
                            if m.get("hand_asset")
                            else "retarget.json"
                        )
                    )
                },
                "urdf_path": str(
                    root / ("hand.urdf" if m.get("hand_asset") else "retarget.urdf")
                ),
            }
            for side in layouts
        },
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
