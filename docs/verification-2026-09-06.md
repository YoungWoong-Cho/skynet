# Hands and live teleoperation verification — 2026-09-06

## Browser and model checks

All seven requested hand types were opened in the actual localhost browser. Twelve supplied, usable left/right variants loaded real mesh geometry. Allegro V4 left remains explicitly unavailable because its pinned source references a missing thumb mesh; LEAP V1 supplies only a right-hand model.

Checked model switching, side selection, joint limits, reset, saved poses across reloads, camera rotation/reset, and the 600 × 850 layout. Inspire's six independent controls move its six linked joints. Shadow's COLLADA finger geometry originally disappeared because embedded material opacity was zero; the viewer now applies the URDF visual material throughout nested mesh geometry. Both Shadow sides render correctly. Both WUJI 2 previews are upright without modifying saved joint coordinates.

Pose export returns validated JSON with model, side and pinned revision. This in-app browser did not confirm a downloaded file and denied the clipboard API. The export dialog therefore exposes the complete JSON and selects it for manual copying when automatic copying fails; the error is visible. File downloading in this browser is not claimed as verified.

Live teleoperation navigation, reload, saved license consent, session submission, reported headset result, status polling and Stop session were exercised through the browser. A late refresh could previously overwrite the result of a start/stop action; mutation responses now update immediately and stale refreshes are discarded. The address column is labeled **Server address**. Ended sessions hide it. The final browser error log was empty.

## GPU session lifecycle

The application submitted job `3770033` on oppy with the pinned native CloudXR 5.0.1 runtime and full DexVerse PickUpStick scene. Logs reached `Teleop Device: handtracking`; this proves scene startup, not an immersive headset connection. Stop session was clicked in the browser. Final results:

- Session `f6cec890-fc17-4b2f-85cd-7b94768c6192`: `STOPPED`, `server_ready: false`, no error.
- Scheduler: `COMPLETED`, exit `0:0`, elapsed `00:05:28`; allocation released.
- Private temporary files are removed after child processes exit. The earlier job `3769579` failed because cleanup assumed the directory contained no nested simulator logs. That failure remains visible in session history.
- Terminal UI status waits for the scheduler result, so a cleanup failure cannot silently appear successful.

## Headset and network boundary

The patched official Apple client was built, installed and launched on the user's visionOS 2.5 headset. The user confirmed the local-network permission prompt appeared and was allowed. The client preserves full connection errors and offers a bounded TCP connection test.

With the server active at `130.207.124.185`, the headset on campus Wi-Fi reported no TCP 48010 response within eight seconds. The same port was reachable from sky2 inside the cluster. No headset signaling connection was observed by the server. The exact firewall/router policy is unknown. Xcode discovery/installation worked on private Wi-Fi but was not reliable on campus Wi-Fi; that is a separate connection from streaming.

Immersive viewing, live hand control and a completed native demonstration remain unverified until a permitted headset-to-GPU route is available. Native live demonstrations also still require conversion/validation before automatic dataset publication, training and evaluation; the UI explicitly marks that path unsupported. The saved local tracking-recording pipeline is a separate existing feature.

## Automated checks

- 53 Python tests passed across live sessions, hands, dashboard, collection, capture processing and browser regressions.
- JavaScript material regression passed for visible nested CAD geometry.
- JavaScript asynchronous UI regression passed for preserving start/stop responses during an older pending refresh.
- Ruff checks passed for the added Python modules, worker, generator and tests.

Reproduce the focused checks with:

```sh
.venv/bin/python -m pytest tests/test_live_xr.py tests/test_hands.py tests/test_dashboard.py tests/test_collection.py tests/test_capture_processing.py tests/test_browser_regressions.py -q
npm run test:hands
npm run test:live
```
