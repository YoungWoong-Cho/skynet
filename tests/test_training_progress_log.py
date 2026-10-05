import hashlib
import json

import pytest

from skynet_app.training_progress_log import read_execution_log


def boundary(tmp_path, old):
    path, receipt = tmp_path / "progress.jsonl", tmp_path / "start.json"
    path.write_bytes(old)
    stat = path.stat()
    receipt.write_text(json.dumps({
        "schema_version": "skynet.training-progress-start/v1",
        "job_id": "123", "restart_count": 2, "path": str(path),
        "start_byte_offset": len(old), "device": stat.st_dev, "inode": stat.st_ino,
        "prefix_bytes": min(64, len(old)), "prefix_sha256": hashlib.sha256(old[-64:]).hexdigest(),
        "at_line_boundary": not old or old.endswith(b"\n"),
    }))
    return path, receipt


def read(path, receipt, **kwargs):
    return read_execution_log(path, receipt, job_id="123", restart_count=2, **kwargs)


def test_old_resets_and_identical_replayed_rows_are_unambiguous(tmp_path):
    row = b'{"step":4,"loss":0.5}\n'
    path, receipt = boundary(tmp_path, row + b'{"step":9}\n' + row)
    assert read(path, receipt) == ""
    with path.open("ab") as stream:
        stream.write(row)
    assert read(path, receipt) == row.decode()
    # No process-local state: a new reader observes the same launch boundary.
    assert read(path, receipt) == row.decode()


@pytest.mark.parametrize("host_specific_fields", [("device",), ("inode",), ("device", "inode")])
def test_shared_file_boundary_survives_different_compute_and_gateway_stat_values(tmp_path, host_specific_fields):
    replayed = b'{"step":4,"loss":0.5}\n'
    old = replayed + b'{"step":100,"loss":0.2}\n' + replayed + b'{"step":99,"loss":0.3}\n'
    path, receipt = boundary(tmp_path, old)
    payload = json.loads(receipt.read_text())
    for key in host_specific_fields:
        payload[key] += 1000  # Receipt was captured by a different NFS client.
    receipt.write_text(json.dumps(payload))
    assert read(path, receipt) == ""  # No new producer writes yet.
    with path.open("ab") as stream:
        stream.write(replayed)
    assert read(path, receipt) == replayed.decode()
    assert read(path, receipt, max_bytes=len(replayed), contains='"loss"') == replayed.decode()


def test_shared_file_partial_line_boundary_is_retained_across_hosts(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":')
    payload = json.loads(receipt.read_text())
    payload["device"] += 1
    receipt.write_text(json.dumps(payload))
    with path.open("ab") as stream:
        stream.write(b'1}\n{"step":2}\n{"step":3')
    assert read(path, receipt) == '{"step":2}\n'


def test_replaced_file_with_preserved_prefix_keeps_logical_boundary(tmp_path):
    old = b'{"old":1}\n'
    path, receipt = boundary(tmp_path, old)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(old + b'{"step":2}\n')
    replacement.replace(path)
    assert read(path, receipt) == '{"step":2}\n'


@pytest.mark.parametrize("missing_at_launch", [False, True])
def test_zero_offset_reads_new_rows_regardless_of_file_identity(tmp_path, missing_at_launch):
    path, receipt = boundary(tmp_path, b"")
    payload = json.loads(receipt.read_text())
    payload.update(device=None if missing_at_launch else payload["device"] + 1,
                   inode=None if missing_at_launch else payload["inode"] + 1)
    receipt.write_text(json.dumps(payload))
    path.write_bytes(b'{"step":1}\n')
    assert read(path, receipt) == '{"step":1}\n'


def test_truncation_below_saved_offset_does_not_skip_new_rows(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":1}\n' * 30)
    path.write_bytes(b'{"step":2}\n')
    assert read(path, receipt) == '{"step":2}\n'


@pytest.mark.parametrize("replace_file", [False, True])
def test_writer_truncation_or_replacement_starts_a_new_file_generation(tmp_path, replace_file):
    path, receipt = boundary(tmp_path, b'{"old":1}\n')
    if replace_file:
        path.unlink()
    new = b'{"step":2,"loss":0.123456789}\n'
    path.write_bytes(new)  # Longer than the old prefix, including the truncate case.
    assert read(path, receipt) == new.decode()


def test_partial_old_line_is_skipped_and_partial_new_line_is_not_consumed(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":')
    with path.open("ab") as stream:
        stream.write(b'1}\n{"step":2}\n{"step":3')
    assert read(path, receipt) == '{"step":2}\n'


def test_tail_is_bounded_and_never_parses_a_cut_line(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":1}\n')
    with path.open("ab") as stream:
        stream.write(b'{"step":1}\n{"step":2}\n{"step":3}\n')
    assert read(path, receipt, max_bytes=26) == '{"step":2}\n{"step":3}\n'
    assert read(path, receipt, lines=1) == '{"step":3}\n'


def test_required_receipt_not_yet_present_does_not_expose_prior_log(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":1}\n')
    receipt.unlink()
    with pytest.raises(FileNotFoundError, match="boundary is not ready"):
        read(path, receipt)
    assert read(path, receipt, required=False) == '{"old":1}\n'


def test_receipt_cannot_be_borrowed_from_another_job_or_restart(tmp_path):
    path, receipt = boundary(tmp_path, b'{"old":1}\n')
    payload = json.loads(receipt.read_text())
    payload["restart_count"] = 1
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        read(path, receipt)
