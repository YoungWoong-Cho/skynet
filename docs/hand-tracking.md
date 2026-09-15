# Hand tracking in live collection

Data → Collect → Retargeting selects a solver before starting a session. DexPilot is the default. The selection cannot change during an active session. Hand models and their coordinate contract come from the [environment-independent hand bundles](hand-assets.md), including six wrist axes, named finger joints, fingertip links, and separate origins for each hand. Both methods use these same physical models and the existing DexVerse simulator and headset input.

With DexPilot, the existing `SimpleAbsoluteRetargeter` and finger solver are unchanged. Start begins the episode; it does not turn the current wrist pose into a relative zero. Finger and translation commands remain identical to DexVerse's output. The shared wrist wrapper selects equivalent continuous Euler angles so position-controlled joints do not spin backward at the ±180° boundary. Missing origins or incomplete finger mappings stop startup with an error.

## Vector Wrist Joint

The optional method uses `VectorWristJointOptimizerV2` from [Mingrui-Yu/retargeting](https://github.com/Mingrui-Yu/retargeting), pinned to `3846d3fa207165bb0d498145aac8b885a28ea923` with utils submodule `2d8dc1a5abf5899069f9ec73c13de73674f4c897`. Its core optimizer runs directly; ROS and a real robot controller are not required.

DexVerse supplies absolute wrist coordinates and canonical 21-point hand observations. The new solver jointly adjusts the six wrist axes and the physical hand's independent finger joints. It replaces only the finger-solver call and final joint command generation. Skynet derives the robot/target configuration from the selected Hands bundle; it does not substitute the upstream Panda/Shadow model. Coupled/mimic joints are currently rejected before a session is created; use DexPilot for those hands.

The upstream full-method objective and pinch thresholds are retained. Skynet's floating-wrist profile uses no wrist position regularization toward zero. For Shadow, adapter version 2 maps the pinned upstream Shadow position weights through the canonical bundle's source joint names, preserving their meaning when joint order or side prefixes differ. Other hands use no absolute finger-position penalty until a calibrated profile exists; all hands retain their physical joint limits and wrist temporal weight `0.1`. Adapter version 4 uses finger temporal weight `0.001` for WUJI 2 and retains `0.01` for the other hands, with a shared `0.016 s` NLopt budget across the hands. The WUJI override reduces the penalty on rapid finger movement; old session configurations without overrides keep their saved weight. The former uniform `0.5` finger-position penalty suppressed flexion and is no longer generated. The former 8 ms cutoff repeatedly returned time-limited intermediate poses after rapid finger movement. This budget limits optimization, not the entire simulator frame; rendering and headset transport add their own time. The first backward pass is warmed before collection becomes ready. Every episode records `skynet_retargeting`, including the source revision, actual generated profile, solver settings, joint names, and kinematic checksums. Solver failures and invalid tracking stop command generation instead of silently changing methods.

Install on a collection host with its existing simulation Python; no sudo is needed:

```sh
python ops/xr/prepare_retargeting.py \
  --work-root /home/rl2-bonjour/skynet-xr \
  --python /home/rl2-bonjour/skynet-xr/env/bin/python
```

The installer keeps the pinned repository and additional dependencies under `retargeters/vector-wrist-joint/<commit>/`, checks their contents, and leaves the simulation environment's packages unchanged. Session preparation checks this installation before starting Isaac or CloudXR. Another collection host needs its own installation.

Different hand geometry and joint counts still limit how closely finger shapes can match a human hand. Inspire's six actuated joints drive six mechanically linked joints; it cannot independently reproduce every human finger joint.

## Simulation hand materials

The simulator applies explicit URDF colors and opacity over nested CAD materials, matching the Hands viewer's precedence. Shadow thumb DAE files import with fully transparent embedded shading even though the URDF specifies an opaque surface. The adapter binds the declared material at each uniformly colored link's visual root with stronger precedence than descendant mesh bindings. Source assets, mesh transforms, and physical properties remain unchanged. Explicit URDF transparency and mixed-material links are preserved. For visuals without a URDF color, the adapter matches the Hands viewer: constant zero opacity in known CAD shaders is restored to one, retaining other shading inputs and nonzero translucency. This restores the Shadow PST fingertip surfaces. Only the affected visual instance is expanded to author its material override; collision materials and shared CAD assets are untouched.

`tests/test_hand_materials.py` covers material resolution and, with OpenUSD installed, checks actual binding precedence on instance meshes, idempotence, explicit translucency, and preservation of physics and collision geometry.

## Inspire conversion

The original GLB meshes display correctly in the Hands page. Isaac's URDF conversion misplaced their root node transforms. Simulation bundles now bake those transforms into vertices while preserving materials, leaving the source models unchanged.

The installed Isaac Lab option `convert_mimic_joints_to_normal_joints` is passed directly to the importer's `parse_mimic`; it must be true for models with mimic joints. The adapter verifies the imported relationships, preserves URDF joint limits, converts offset sign/radians to PhysX's constraint equation/degrees, and disables the importer's weak spring compliance. This preserves the URDF's mechanical linkage without changing retargeting.

## Verification

`ops/xr/check_following.py` exercises every available hand against actual simulator bodies. Pass `--robot`, `--result`, and `--bundle` for imported hands. `--images` captures open and closed poses. Set `SKYNET_DEXVERSE_RECORDER` to the pinned `scripts/record_demos.py`; run with the workstation's GPU session lock while collection is stopped. Require a result with `status: GPU_CHECKED`: Isaac shutdown can mask Python exceptions in the process exit code.

`ops/xr/check_collection.py` additionally checks manual Start, tracking interruption, two saved episodes, and reset with the same production collection loop. Automated results and their distinction from headset confirmation are recorded in `validation/hand-following.json`.

`ops/xr/check_retargeting.py` runs both production command paths against actual PhysX joints with synthetic OpenXR poses, without creating recordings. Pass the pinned `--bundle`, a Vector Wrist Joint `--config` JSON, and `--result`. Use the same environment variables and GPU lock as `check_following.py`. It requires each finger to curl and reopen, measures fingertip motion in the hand control frame so wrist movement cannot count as finger tracking, and checks final target/actual angles. Add `--rapid` to replace the slow ramps with instantaneous close/open steps and retain every target, actual joint pose, and frame duration. Settled-pose checks alone do not establish rapid tracking latency. The complete measurements are retained on failure. This verifies simulator integration; it does not establish headset feel, task success, or compatibility with every hand.

## Live response and progress

Shadow and WUJI 2 finger drives use `armature=0.001`, separately from the virtual wrist's `0.01`; copying the wrist value to light finger joints added avoidable tracking delay and overshoot. WUJI 2 finger velocity caps come from each canonical URDF joint (8.11–13.5 rad/s), replacing the blanket 5 rad/s cap. Shadow keeps its existing 5 rad/s cap. Stiffness, damping, effort caps, canonical meshes, masses, and physical inertias are unchanged. Runtime bundle hashes pin this setting for new sessions. Other hand models retain their previous actuator settings until their response is measured.

Live status HTTP requests return saved progress and queue one background remote check per session. Startup and progress updates use the live-collection lock and an atomic PostgreSQL transaction without taking the unrelated repository-wide write lock. Reading license acceptance does not take a write lock. Browser request timeouts show a retrying status instead of continuing to claim an old startup step is in progress.

Generated simulation bundles are cached outside the checkout (`~/Library/Caches/skynet/<checkout>/simulation-hands` on macOS, `$XDG_CACHE_HOME/skynet/<checkout>/simulation-hands` or `~/.cache/...` on Linux). Creating `runtime.py` inside `data/simulation-hands` made a bare `uvicorn --reload` treat a new hand as a source edit, interrupting the API and delaying startup by roughly 100 seconds. New bundles no longer trigger this reload; pinned bundles from the old location remain readable for existing recordings.
