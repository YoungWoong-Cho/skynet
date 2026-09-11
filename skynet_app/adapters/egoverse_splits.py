"""Episode membership shared by preparation and the frozen native runtime."""

OVERFIT_MODE = "single_episode_overfit"


def validate_split(split, episode_count):
    train, validation = split["train"], split["validation"]
    if any(type(i) is not int for i in train + validation):
        raise ValueError("Episode indices must be integers")
    if split.get("mode") == OVERFIT_MODE:
        if episode_count != 1 or train != [0] or validation != [0]:
            raise ValueError("Single-episode overfit must reuse exactly one episode")
        return
    if split.get("mode") not in {None, "held_out"} or (
        not train or not validation
        or sorted(train + validation) != list(range(episode_count))
    ):
        raise ValueError("Every episode must belong to exactly one training or validation split")
