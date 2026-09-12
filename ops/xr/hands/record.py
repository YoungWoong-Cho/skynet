"""Run the unmodified pinned recorder with one imported hand registered in-process."""

import json
import os
from pathlib import Path
import runpy
from runtime import install, validate_environment

if __name__ == "__main__":
    recorder = Path(os.environ["SKYNET_DEXVERSE_RECORDER"])
    namespace = runpy.run_path(str(recorder), run_name="skynet_upstream_recorder")
    app = namespace["simulation_app"]
    try:
        manifest = install(Path(__file__).resolve().parent)
        if namespace["args_cli"].robot_type != manifest["robot"]:
            raise ValueError("Selected robot does not match the prepared hand bundle")
        globals_ = namespace["main"].__globals__
        original_create = globals_["create_environment"]

        def create(*args, **kwargs):
            env = original_create(*args, **kwargs)
            validate_environment(env, manifest)
            return env

        globals_["create_environment"] = create
        recorder_cls = globals_["TrajectoryPickleRecorder"]
        original_init = recorder_cls.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._metadata["skynet_hand"] = {
                key: manifest[key]
                for key in (
                    "robot",
                    "hand_key",
                    "side",
                    "source_revision",
                    "digest",
                    "source_names",
                    "mimic_joints",
                    "hand_asset",
                    "hand_order",
                    "units",
                    "wrist_rotation_order",
                    "hands",
                )
                if key in manifest
            }
            self._metadata["skynet_hand"]["action_joint_names"] = (
                manifest["wrist_joints"] + manifest["finger_joints"]
            )

        recorder_cls.__init__ = init
        print(
            json.dumps(
                {
                    "skynet_hand": manifest["name"],
                    "source_revision": manifest["source_revision"],
                    "action_dimension": manifest["action_dimension"],
                }
            ),
            flush=True,
        )
        namespace["main"]()
    except Exception as exc:
        print("SKYNET_HAND_ERROR: " + str(exc), flush=True)
        raise
    finally:
        app.close()
