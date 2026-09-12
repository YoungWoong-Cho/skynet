# Hand tracking in live collection

Every selectable hand uses DexVerse's `SimpleAbsoluteRetargeter` with the standard DexPilot finger solver. Start begins the episode; it does not turn the current wrist pose into a relative zero. The robot follows the tracked wrist's world pose. Hand models and their coordinate contract are defined in the [environment-independent hand bundles](hand-assets.md). DexVerse supplies the retargeter and simulator configuration. Each robot supplies its joint order, fingertip links, and wrist origin, including separate origins for bimanual Shadow.

Skynet does not replace DexVerse's finger references, canonical hand conversion, or finger solver. The simulator wrapper only selects equivalent continuous wrist Euler angles so position-controlled joints do not spin backward at the ±180° boundary. Finger and translation commands remain identical to DexVerse's output. Missing origins or incomplete finger mappings stop startup with an error.

Different hand geometry and joint counts still limit how closely finger shapes can match a human hand. Inspire's six actuated joints drive six mechanically linked joints; it cannot independently reproduce every human finger joint.

## Inspire conversion

The original GLB meshes display correctly in the Hands page. Isaac's URDF conversion misplaced their root node transforms. Simulation bundles now bake those transforms into vertices while preserving materials, leaving the source models unchanged.

The installed Isaac Lab option `convert_mimic_joints_to_normal_joints` is passed directly to the importer's `parse_mimic`; it must be true for models with mimic joints. The adapter verifies the imported relationships, preserves URDF joint limits, converts offset sign/radians to PhysX's constraint equation/degrees, and disables the importer's weak spring compliance. This preserves the URDF's mechanical linkage without changing retargeting.

## Verification

`ops/xr/check_following.py` exercises every available hand against actual simulator bodies. Pass `--robot`, `--result`, and `--bundle` for imported hands. `--images` captures open and closed poses. Set `SKYNET_DEXVERSE_RECORDER` to the pinned `scripts/record_demos.py`; run with the workstation's GPU session lock while collection is stopped. Require a result with `status: GPU_CHECKED`: Isaac shutdown can mask Python exceptions in the process exit code.

`ops/xr/check_collection.py` additionally checks manual Start, tracking interruption, two saved episodes, and reset with the same production collection loop. Automated results and their distinction from headset confirmation are recorded in `validation/hand-following.json`.
