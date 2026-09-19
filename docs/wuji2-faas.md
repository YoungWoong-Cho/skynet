# WUJI Hand 2 (Beta 2) → UniDex FAAS

This is a **WUJI2-specific anatomical mapping**, not an alias of the original
UniDex `Wuji` asset. It uses the existing FAAS82 codec and original slot layout.
Neither the simulator asset nor recorded commands are changed.

## Verification scope

VERIFIED means the named joints, positive motion, declared local reference,
wrist frame and codec reversibility are checked against the registered asset.
It does not establish identical fingertip poses to upstream Wuji, calibration to
a pretrained checkpoint, or cross-hand generalization performance. Those last
claims require training and held-out evaluation.

## Sources

- Current geometry: `config/action_representations/unidex-faas-v1/assets/skynet_wuji_2_right.urdf`, bound by the spec/capture SHA-256.
- Functional reference: UniDex commit `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`, `src/assets/utils/hand_utils.json`.
- Reference geometry: `HandAdapter/urdf/H2o/Wuji/right/main.urdf`, SHA-256 `94c06247d1b0250e9c12c43ed9957233c414c765903f08a00c8a883186fd7c66`.
- Original mapping: `src/dataset/base.py`, `_apply_scale_shift` then `_apply_action_map`. This applies declared per-model joint coordinates, without solving a common fingertip pose.

## Finger mapping

`faas[18 + slot] = native_joint_radians * scale + offset`. Slots are zero-based
within the right-hand finger block; the prefix contains the two wrists.

| Local joints in control order | Slots | Scales | Offsets |
| --- | --- | --- | --- |
| Thumb CMC flex, CMC abd, MCP, IP | 1, 2, 3, 4 | +1, +1, +1, +1 | 0, 0, 0, 0 |
| Index MCP flex, MCP abd, PIP, DIP | 7, 6, 8, 9 | +1, −1, +1, +1 | 0, 0, 0, 0 |
| Middle MCP flex, MCP abd, PIP, DIP | 12, 11, 13, 14 | +1, −1, +1, +1 | 0, 0, 0, 0 |
| Ring MCP flex, MCP abd, PIP, DIP | 17, 16, 18, 19 | +1, −1, +1, +1 | 0, 0, 0, 0 |
| Pinky MCP flex, MCP abd, PIP, DIP | 22, 21, 23, 24 | +1, −1, +1, +1 | 0, 0, 0, 0 |

`source_joint` names refer to the actual WUJI2 asset. A separate
`upstream_slot_reference` records functional correspondence, not equal geometry.
Upstream Wuji non-thumb MCP flex uses −1 and abduction +1. Copying those signs
would be wrong: WUJI2 needs +1 flex and −1 abduction. With corrected signs,
neutral physical-axis dot products against the upstream functional reference are
0.992–1.000 for non-thumb joints and 0.892–0.989 for thumb joints. Independent
joint sweeps check the actual fingertip movement, not just invertibility.

## Explicit local zero reference

WUJI2's native URDF q=0 is its **declared local rest reference**, not an assertion
of the same pose as another morphology. This follows the original per-model
coordinate convention: Allegro, LEAP, Wuji and Xhand all use zero offsets despite
distinct anatomy. Other explicit corrections include Shadow THJ5 +0.52 and
Inspire intermediate joints −0.04545; they are not transferable to WUJI2.

At q=0 the non-thumb chains are extended with their designed base splay. Thumb
MCP/IP have zero signed flexion projected into their physical flexion planes.
The thumb CMC mounting rotations and out-of-plane MCP mounting angle remain
part of the asset geometry. Both CMC zeros explicitly mean the WUJI2 mounted
rest configuration. They are not a geometrically fitted upstream open-hand pose.

Thus all offsets are intentionally zero under this versioned local convention.
No demonstration statistics, range midpoints, limit rescaling, or pose fitting
are used. A future alternative cross-hand zero calibration must change the spec
fingerprint and regenerate dependent FAAS streams; it must not silently reinterpret
old data. Declaring this convention does not guarantee checkpoint transfer.

## Own wrist frame

Retain `h_r_wrist` origin, with no borrowed translation. In its coordinates,
canonical +X is palmar (+Y), +Y radial (+X), +Z distal (−Z):

```text
Columns are canonical axes in the native physical wrist:
0  1  0
1  0  0
0  0 -1
```

This proper rotation uses the same functional palmar/radial/distal convention
as upstream Wuji in a different physical basis. The codec also composes WUJI2's
own fixed `skynet_alignment`. Relative to `skynet_palm`, the result is
`[[0,0,1],[0,1,0],[-1,0,0]]`, zero translation. No upstream `base_joint`, camera
pose, link length, or wrist offset is copied.

## Tests

`tests/test_wuji2_faas.py` independently parses the current URDF and tests all
20 joint sweeps, wrist geometry, checksum, declared rest pose, thumb projected
flexion zeros and controller-command round trips. Reference axis/tangent values
are derived from the pinned upstream URDF at q=0: `axis × (tip − joint_origin)`,
expressed using the official right-hand HAND_TRANSFORMS and FAAS signs. They are
independent constants, not recomputed from the new mapping. Tolerances allow
different morphologies; no exact geometric identity is claimed.

CPU mapping checks do not establish GPU observation generation, actual model
training, simulator controller rollout, or unseen-hand performance.
