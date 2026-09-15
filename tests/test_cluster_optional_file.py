"""Optional file reads distinguish confirmed absence from transport/read errors."""
import subprocess

import pytest

import skynet_app.cluster_runtime as runtime
from skynet_app.cluster_runtime import ClusterClient, ClusterError


@pytest.fixture
def local_client(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'WORK_ROOT', str(tmp_path.resolve()))
    client = ClusterClient(('sky2',))
    def ssh(host, command, *, stdin=None, timeout=30):
        # The remote command uses GNU stat; emulate that one operation on macOS.
        command = 'stat() { wc -c < "$3"; };\n' + command
        result = subprocess.run(['bash', '-c', command], text=True, capture_output=True, input=stdin)
        if result.returncode:
            raise ClusterError(result.stderr)
        return result.stdout
    monkeypatch.setattr(client, 'ssh', ssh)
    return client


@pytest.mark.parametrize('content', ['', 'SKYNET_FILE_MISSING\nreal result\n'])
def test_optional_file_preserves_empty_and_marker_content(local_client, tmp_path, content):
    path = tmp_path.resolve() / "result 'quoted name'.json"
    path.write_text(content)
    assert local_client.read_optional_file(str(path), 'sky2') == ('sky2', content)
    path.unlink()
    assert local_client.read_optional_file(str(path), 'sky2') == ('sky2', None)


def test_optional_file_read_limit_is_an_error_not_missing(local_client, tmp_path):
    path = tmp_path.resolve() / 'large.json'
    path.write_text('x' * 1025)
    with pytest.raises(ClusterError):
        local_client.read_optional_file(str(path), 'sky2', max_bytes=1024)


def test_optional_file_transport_and_protocol_errors_are_not_missing(local_client, tmp_path, monkeypatch):
    path = str(tmp_path.resolve() / 'missing.json')
    def offline(*args, **kwargs):
        raise ClusterError('network interrupted')
    monkeypatch.setattr(local_client, 'ssh', offline)
    with pytest.raises(ClusterError):
        local_client.read_optional_file(path, 'sky2')
    monkeypatch.setattr(local_client, 'ssh', lambda *args, **kwargs: 'unframed reply')
    with pytest.raises(ClusterError):
        local_client.read_optional_file(path, 'sky2')
