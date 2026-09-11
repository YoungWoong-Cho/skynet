"""Task assignment, canonical merging, and resume contracts for parallel rollouts."""

import copy
import json
from pathlib import Path
import pytest

SUPPORT = Path(__file__).parents[1] / "skynet_app/adapters"


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.syspath_prepend(str(SUPPORT))
    import evaluation_workers

    return evaluation_workers


@pytest.fixture
def context(tmp_path):
    return dict(
        run_id="run",
        tasks=["cube", "stick"],
        seeds=[0, 7],
        episodes_per_task=3,
        parallelism=2,
        checkpoint=dict(path="checkpoint", sha256="a" * 64),
        suite=dict(name="dexverse_recorded", version="tasks-v2"),
        evaluator=dict(adapter="isaac_lab", version="1"),
        policy={},
        result_path=str(tmp_path / "result.json"),
        progress_path=str(tmp_path / "progress.jsonl"),
    )


def test_assignments_cover_all_tasks_seeds_without_reusing_episodes(workers, context):
    units = workers.worker_contexts(context)
    assigned = [
        (unit["tasks"][0], seed, index)
        for unit in units
        for seed, index in unit["episode_assignments"]
    ]
    assert len(assigned) == 12 and len(set(assigned)) == 12
    assert set(assigned) == set(workers.expected_keys(context))
    assert units == workers.worker_contexts(copy.deepcopy(context))
    assert all(unit["parallelism"] == 1 and len(unit["tasks"]) == 1 for unit in units)
    assert len({unit["progress_path"] for unit in units}) == 4
    assert all(
        Path(context["result_path"]).parent / "videos" in Path(u["video_path"]).parents
        for u in units
    )
    context["parallelism"] = 8
    context["episodes_per_task"] = 1
    with pytest.raises(ValueError, match="workers"):
        workers.worker_contexts(context)


def test_merged_success_rate_uses_episodes_not_average_of_workers(workers, context):
    observed = {
        key: dict(
            task=key[0],
            seed=key[1],
            episode_index=key[2],
            success=key[0] == "cube",
            video_path="/video.mp4",
            status="SUCCEEDED",
        )
        for key in workers.expected_keys(context)
    }
    result = workers.merged_result(context, observed)
    assert result["aggregate"][0]["mean"] == 0.5
    assert result["aggregate"][0]["sample_count"] == 12
    assert [row["mean"] for row in result["aggregate"][1:]] == [1, 0]
    observed.pop(next(iter(observed)))
    with pytest.raises(ValueError, match="every requested"):
        workers.merged_result(context, observed)


def test_progress_survives_partial_worker_write_and_checks_identity(workers, context):
    from dexverse_evaluation import identity

    units = workers.worker_contexts(context)
    unit = units[0]
    seed, index = unit["episode_assignments"][0]
    video = Path(unit["video_path"]) / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"mp4")
    episode = dict(
        task=unit["tasks"][0],
        seed=seed,
        episode_index=index,
        success=True,
        status="SUCCEEDED",
        video_path=str(video),
    )
    path = Path(unit["progress_path"])
    path.parent.mkdir(parents=True)
    row = dict(identity=identity(unit), status="SUCCEEDED", episode=episode)
    path.write_text(json.dumps(row) + '\n{"interrupted":')
    observed = workers.completed_results(context, units)
    assert list(observed.values()) == [episode]
    workers.publish_progress(context, observed)
    record = json.loads(Path(context["progress_path"]).read_text())
    assert record["kind"] == "episode_observed" and record["total"] == 12
    row["identity"] = "foreign"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="different checkpoint"):
        workers.completed_results(context, units)


def test_parallel_capability_preserves_serial_historical_manifests():
    from skynet_app.adapters import EvaluationAdapterMetadata, builtin_adapter_manifests

    old = EvaluationAdapterMetadata(environment="sim")
    assert "maximum_parallelism" not in old.model_dump()
    for manifest in builtin_adapter_manifests():
        if manifest.slug in {"xpolicylab-dp", "xpolicylab-act"}:
            assert manifest.evaluations[0].maximum_parallelism == 8
            assert manifest.capabilities.maximum_gpus == 8
            assert "evaluation_workers.py" in manifest.evaluations[0].command.argv[1]


def test_running_episode_transitions_to_complete_without_counting_twice(
    workers, context
):
    from dexverse_evaluation import append_progress, identity

    units = workers.worker_contexts(context)
    unit = units[0]
    seed, index = unit["episode_assignments"][0]
    path = Path(unit["progress_path"])
    path.parent.mkdir(parents=True)
    episode = dict(
        task=unit["tasks"][0],
        seed=seed,
        episode_index=index,
        status="RUNNING",
        metrics={"worker_index": 0},
    )
    append_progress(
        path, dict(identity=identity(unit), status="RUNNING", episode=episode)
    )
    assert list(workers.observed_results(context, units).values()) == [episode]
    assert workers.completed_results(context, units) == {}
    video = path.with_suffix(".mp4")
    video.write_bytes(b"mp4")
    with path.open("a") as stream:
        stream.write('{"interrupted":')
    episode.update(status="SUCCEEDED", success=False, video_path=str(video))
    append_progress(
        path, dict(identity=identity(unit), status="SUCCEEDED", episode=episode)
    )
    assert list(workers.completed_results(context, units).values()) == [episode]


def test_simulator_executes_single_episode_without_worker_assignment(workers, context):
    from dexverse_evaluation import assigned_episodes
    single = dict(context, seeds=[42], episodes_per_task=1,
                  recorded_episode_sources=[{"path": "/recorded/episode.pkl"}])
    assert assigned_episodes(single) == {(42, 0)}
    for unit in workers.worker_contexts(context):
        assert assigned_episodes(unit) == {tuple(pair) for pair in unit["episode_assignments"]}
    with pytest.raises(ValueError, match="assignments"):
        assigned_episodes(dict(single, episode_assignments=[[42, 1]]))
