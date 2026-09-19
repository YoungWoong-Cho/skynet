# Sharpa Wave → UniDex FAAS

## Original UniDex versus Skynet

**Original UniDex** at `97d869e0f2d1ec0372cd3cdf28dde66b4e3f216d`
has no Sharpa entry. Its right-hand FAAS finger block has 32 slots; the wrist
and finger layout remains unchanged.

**Skynet's Sharpa extension** binds all 22 independent finger joints to those
existing slots using the actual registered Sharpa URDF. It does not alias
Sharpa to Wuji or Shadow, replace simulation assets, or rewrite recorded commands.
The contract retains the asset and controller hashes and is fingerprinted.

## Mapping

`faas[18 + slot] = native_joint_radians * scale + offset`.
Slots below are zero-based within the right-hand finger block.

| Sharpa joints | Slots | Scale | Functional reference in original UniDex |
| --- | --- | --- | --- |
| Thumb CMC FE, CMC AA, MCP FE, MCP AA, IP | 1, 2, 3, 26, 4 | +1 for all | Wuji F1J1/F1J2/F1J3/F1J4; Shadow THJ3 for MCP AA |
| Index MCP FE, MCP AA, PIP, DIP | 7, 6, 8, 9 | +1, −1, +1, +1 | Shadow FFJ3/FFJ4/FFJ2/FFJ1 |
| Middle MCP FE, MCP AA, PIP, DIP | 12, 11, 13, 14 | +1, −1, +1, +1 | Shadow MFJ3/MFJ4/MFJ2/MFJ1 |
| Ring MCP FE, MCP AA, PIP, DIP | 17, 16, 18, 19 | +1, −1, +1, +1 | Shadow RFJ3/RFJ4/RFJ2/RFJ1 |
| Pinky CMC, MCP FE, MCP AA, PIP, DIP | 25, 22, 21, 23, 24 | +1, +1, −1, +1, +1 | Shadow LFJ5/LFJ3/LFJ4/LFJ2/LFJ1 |

All offsets explicitly use **Sharpa native URDF q=0** as the local rest.
Thumb mounting angles and pinky metacarpal geometry remain part of that asset.
This is not a fitted common open-hand pose, an upstream zero calibration, or a
normalization by joint limits. In particular, Shadow THJ5's offset is not copied.

The physical `h_right_hand_C_MC` origin is retained. Its native axes are already
canonical +X palmar, +Y radial, +Z distal. The codec composes the actual registered
`skynet_alignment` with this identity frame; no upstream wrist translation,
retargeting base pose or camera extrinsic is borrowed.

## Geometry checks and scope

`tests/test_sharpa_faas.py` parses the registered URDF independently of the
production forward-kinematics code and sweeps each of the 22 joints. It verifies
slot completeness, physical axis and fingertip motion signs, unaffected other
fingers, own wrist origin, rest convention, checksum and reversible commands.
Reference vectors are derived from the pinned original UniDex Wuji and Shadow
URDFs with their official right-hand HAND_TRANSFORMS and FAAS signs.

Non-thumb flexion/abduction axes and tangents agree with the Shadow functional
references to dot product >0.9999. Thumb axis/tangent comparisons exceed 0.85/0.80.
Sharpa's proximal pinky CMC is a rotary metacarpal with lateral travel; Shadow's
is an oblique hinge. Both produce positive palmar cupping, but their fingertip
tangents are different (dot product about 0.50). Slot 25 records their functional
correspondence, not geometric equivalence. The exact native coordinate remains
reversible; no joint is dropped or forced into another actuator's slot.

`VERIFIED` covers this geometry/coordinate contract. It does not establish
identical fingertip positions, pretrained checkpoint calibration, GPU observation
rendering, controller rollout success, or unseen-hand generalization.
