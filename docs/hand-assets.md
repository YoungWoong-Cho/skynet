# Shared hand assets

`config/hands.json` is the physical hand catalog. Its `floating_hand` section identifies the palm subtree, fingertip frames and the fixed transform to the common control frame. It contains no simulator runtime or teleoperation solver configuration.

`skynet_app/hand_bundles.py` produces immutable `skynet.hand-bundle/v1` bundles at `data/hand-bundles/<robot>/<digest>/`:

- `hand.urdf`: palm and fingers, with a normalized control frame.
- `simulation.urdf`: the same hand with translation X/Y/Z and intrinsic XYZ rotation joints. This is a portable URDF, not a simulator-specific binary.
- `assets/`: pinned meshes and source license files; GLB node transforms are baked consistently.
- `manifest.json`: model revision and checksum, source-to-canonical names, units, joint limits, mimic relationships, per-hand layout, action order and file checksums.

Lengths are meters, angles radians, masses kilograms. The canonical action contract is all hands' wrist axes followed by all hands' independent finger joints. `hand_order`, `wrist_joints` and `finger_joints` are explicit ordered lists; consumers must not infer order from JSON object keys or sort joint names. Mimic joints remain mechanical relationships, not extra commanded actions.

Shadow uses the official model already present in Hands: the palm and its 22 finger joints, without the original forearm or WRJ1/WRJ2 hardware wrist. Six virtual axes give 28 actions. The bimanual assembly composes two pinned hands with independent wrists, yielding 56 actions: right wrist, left wrist, right fingers, left fingers. Original palm/finger masses and inertias are preserved; no DexVerse 10× fingertip adjustment is applied.

## Environment adapters

`skynet_app/simulation_hands.py` builds the current DexVerse adapter at `data/simulation-hands/<robot>/<digest>/`. It pins the canonical `hand_asset.digest`, adds DexPilot configs and a frozen runtime, and keeps simulator conversion, drives, collision exclusions, initial placements and teleoperation configuration outside the physical catalog. Changing these adapter settings changes the adapter digest independently of the physical hand identity.

A new simulator or teleoperation adapter should consume the canonical bundle, convert its geometry as needed, and map its I/O to the ordered manifest contract. It must preserve model provenance and declare its own controller and solver settings. A matching joint count alone is not compatibility evidence.

Live collection and saved Vision Pro tracking processing both build and stage this adapter. New sessions cannot select DexVerse's `floating_shadow_*` robots. The old non-runnable `dexverse-cloudxr` manifest remains a legacy import/schema template; it is not the live launcher. Do not turn that native-format template into a new collection launcher; use the canonical adapter instead.

## Recording compatibility

New recordings retain the physical hand identity, adapter digest, source names, ordered action joints and per-hand layout. Published bundles are immutable and validated before use. The existing `/hands/<robot>/<digest>/` remote layout remains compatible with frozen session references.

Historical recordings keep their recorded native robot or archived adapter. Review accepts known historical native Shadow identities only for reading; those identities do not become new-session choices. Replaying or evaluating an old recording therefore uses its original model and joint convention, not a new model with similarly shaped arrays. Recollect or perform an explicit validated conversion to use the new Shadow model.

## Validation

Unit and asset checks cover exact palm-subtree physical preservation, six-axis contracts, mimic references, corruption rejection, bimanual assembly/action order and historical review. `ops/xr/hands/check_kinematics.py` accepts single and bimanual bundles for real Pinocchio/DexPilot checks. `ops/xr/check_following.py --bundle ...` exercises the actual simulator, retargeter and physical bodies with synthetic tracking. GPU results and headset observations are separate evidence; neither is inferred from unit tests.
