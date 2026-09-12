"""Run an unscored real observation/inference/action cycle before evaluation."""

from policy_contract import validate_observation, validate_actions


def verify_cycle(contract, *, reset, observe, predict, advance, report):
    report({"status": "RUNNING", "phase": "observation_inference_action"})
    try:
        reset()
        observation = observe()
        validate_observation(contract, observation)
        actions = validate_actions(contract, predict(observation))
        advance(actions[0])
        # Reset policy history, random seeds and simulator state before scoring.
        reset()
    except Exception as error:
        report({"status": "FAILED", "error": str(error)})
        raise
    report({"status": "PASSED", "phase": "observation_inference_action"})
