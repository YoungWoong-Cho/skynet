# WUJI Hand 1 → UniDex FAAS

This is a **Skynet extension for the exact registered WUJI1 asset**. It preserves
the original UniDex FAAS82 layout and codec; it does not declare WUJI1 identical
to the original UniDex `Wuji` model. The simulator asset and source PKLs are unchanged.

## Sources and validation scope

- Registered geometry: `config/action_representations/unidex-faas-v1/assets/skynet_wuji_1_right.urdf`, SHA-256 `e4c920a72373ef2e82647786a0880b741a6e54b883ce4cc52646da259309fc15`.
- Original UniDex functional slots: commit `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`, `src/assets/utils/hand_utils.json`.
- Original reference geometry: `HandAdapter/urdf/H2o/Wuji/right/main.urdf`, SHA-256 `94c06247d1b0250e9c12c43ed9957233c414c765903f08a00c8a883186fd7c66`.

VERIFIED covers functional slots, positive anatomical motion, explicit local
rest convention, the wrist frame, asset identity and codec reversibility. It
does not establish identical fingertip poses, pretrained-checkpoint calibration,
or unseen-hand generalization. Those require separate model evaluation.

## Finger coordinates

`faas[18 + slot] = native_joint_radians`. All 20 scales are **+1** and all offsets
are **0**, relative to this asset's native URDF coordinate reference.

| WUJI1 joints | Function | FAAS finger slots |
| --- | --- | --- |
| `right_finger1_joint1..4` | Thumb CMC flex, CMC abduction, MCP, IP | 1, 2, 3, 4 |
| `right_finger2_joint1..4` | Index MCP flex, MCP abduction, PIP, DIP | 7, 6, 8, 9 |
| `right_finger3_joint1..4` | Middle MCP flex, MCP abduction, PIP, DIP | 12, 11, 13, 14 |
| `right_finger4_joint1..4` | Ring MCP flex, MCP abduction, PIP, DIP | 17, 16, 18, 19 |
| `right_finger5_joint1..4` | Pinky MCP flex, MCP abduction, PIP, DIP | 22, 21, 23, 24 |

The registered simulation prefixes each source joint name with `h_`.
The slot order follows the original UniDex functional correspondence. The signs
come from WUJI1's own axes: copying upstream Wuji's negative non-thumb MCP-flex
signs, or WUJI2's negative MCP-abduction signs, would be incorrect.

Independent joint sweeps compare each positive FAAS motion with the pinned
upstream geometry in its official wrist frame. Across all 20 joints, normalized
axis dot products exceed 0.99998 and fingertip-motion dot products exceed 0.9998.
Geometry and limits still differ, so this is an anatomical extension rather than
an upstream asset alias.

Native q=0 is an explicit coordinate reference, not an executable pose or a
claim of equal fingertip positions across hands. In particular, WUJI1 thumb CMC
joint1 has a lower limit of 0.0475 rad. Conversion preserves recorded values and
does not offset them to that limit, fit demonstrations, or normalize joint ranges.
Execution-time limit handling remains separate from lossless encoding.

## Wrist frame

Use WUJI1's own `h_right_palm_link` origin and identity local transform:
**+X palmar, +Y radial, +Z distal**. Compose the existing `skynet_alignment`
rotation when converting the virtual wrist, giving
`[[0,0,1],[0,1,0],[-1,0,0]]` relative to `skynet_palm` and zero translation.
No original Wuji `base_joint` pose, wrist offset, camera pose, or link length is copied.

## Checks

`tests/test_wuji1_faas.py` independently parses the asset URDF and checks all
20 joint sweeps against pinned upstream axis/tangent constants, wrist axes,
finger ordering, the versioned checksum, local zero convention and controller
command round trips, including out-of-limit targets without clipping.

The existing 51 WUJI1 source recordings are also checked read-only for exact
capture contracts, source hashes, complete-frame FAAS encoding, camera-relative
chunk anchoring, controller command round trips, and finite native normalization.
These CPU checks do not render observations or run training.
