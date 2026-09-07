"""Visible wrist/knuckle targets using the same geometry as the alignment gate."""

import numpy as np

from anatomy import palm_frame

ANCHORS = [0, 5, 17]  # Wrist, index knuckle, pinky knuckle in the 21-point hand.
TOLERANCE = 0.025


def alignment_points(points, side, wrist_target):
    """Preserve the person's palm dimensions while removing its pose.

    These are alignment targets, not robot finger joints. Using the person's
    knuckles avoids asking differently sized human and robot hands to coincide.
    """
    p = np.asarray(points, dtype=np.float64)
    frame = palm_frame(p, side)
    tracked = p[ANCHORS]
    targets = (tracked - p[0]) @ frame + np.asarray(wrist_target)
    return tracked, targets, np.linalg.norm(tracked - targets, axis=1)


def ring_mesh(radius=TOLERANCE, thickness=0.0015, segments=40, tube_segments=6):
    """Three perpendicular hoops make a depth target visible from any angle."""
    vertices, faces = [], []
    for axes in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        offset = len(vertices)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            for j in range(tube_segments):
                tube = 2 * np.pi * j / tube_segments
                r = radius + thickness * np.cos(tube)
                point = np.zeros(3)
                point[list(axes)] = [
                    r * np.cos(angle),
                    r * np.sin(angle),
                    thickness * np.sin(tube),
                ]
                vertices.append(point)
                faces.append(
                    [
                        offset + i * tube_segments + j,
                        offset + ((i + 1) % segments) * tube_segments + j,
                        offset
                        + ((i + 1) % segments) * tube_segments
                        + (j + 1) % tube_segments,
                        offset + i * tube_segments + (j + 1) % tube_segments,
                    ]
                )
    return np.asarray(vertices), np.asarray(faces)


def _spawn_ring(prim_path, cfg):
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    vertices, faces = ring_mesh()
    mesh.CreatePointsAttr(vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    material_path = prim_path + "/Material"
    cfg.visual_material.func(material_path, cfg.visual_material)
    sim_utils.bind_visual_material(prim_path, material_path)
    return mesh.GetPrim()


class AlignmentGuide:
    def __init__(self, sides):
        import isaaclab.sim as sim_utils
        from dexverse.visual_purpose import hide_marker_from_cameras
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg

        def material(color):
            return sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, emissive_color=color
            )

        self.markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/collection_alignment",
                markers={
                    "dot": sim_utils.SphereCfg(
                        radius=0.006, visual_material=material((1.0, 1.0, 1.0))
                    ),
                    "target": ShapeCfg(
                        func=_spawn_ring,
                        visual_material=material((1.0, 0.65, 0.04)),
                    ),
                    "matched": ShapeCfg(
                        func=_spawn_ring,
                        visual_material=material((0.12, 1.0, 0.25)),
                    ),
                },
            )
        )
        hide_marker_from_cameras(self.markers)
        self.sides = sides
        self.last_targets = {}
        self.visible = False
        self.markers.set_visibility(False)

    def update(self, points, wrist_targets, show):
        positions, indices = [], []
        if show:
            for side in self.sides:
                data = points.get(side) if points else None
                if data is not None:
                    try:
                        tracked, targets, errors = alignment_points(
                            data, side, wrist_targets[side]
                        )
                    except ValueError:
                        data = None
                    else:
                        self.last_targets[side] = targets
                        positions.extend(tracked)
                        indices.extend([0] * 3)
                        positions.extend(targets)
                        indices.extend(np.where(errors <= TOLERANCE, 2, 1).tolist())
                if data is None:
                    # Leave the targets visible when tracking is lost, but never
                    # leave apparently tracked dots frozen in their last pose.
                    targets = self.last_targets.get(side)
                    if targets is None:
                        sign = 1 if side == "right" else -1
                        targets = np.asarray(wrist_targets[side]) + [
                            [0, 0, 0],
                            [0.08, sign * 0.03, 0],
                            [0.07, -sign * 0.04, 0],
                        ]
                    positions.extend(targets)
                    indices.extend([1] * 3)
        visible = bool(positions)
        if visible != self.visible:
            self.markers.set_visibility(visible)
            self.visible = visible
        if visible:
            self.markers.visualize(
                translations=np.asarray(positions), marker_indices=indices
            )
