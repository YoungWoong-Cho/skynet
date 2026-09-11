"""In-memory cluster transport for collection tests; no SSH or disk payloads."""
import hashlib
from skynet_app.local_capture_storage import CaptureStorage


class MemoryStorage(CaptureStorage):
    def __init__(self):
        super().__init__(cluster=self)
        self.files = {}
        self.published = []
        self.staged = []

    def publish(self, data, digest):
        assert hashlib.sha256(data).hexdigest() == digest
        path = self.path(digest)
        if path in self.files:
            assert self.files[path] == data
        self.files[path] = bytes(data)
        self.published.append(digest)
        self.verify(digest, len(data))
        return path

    def verify(self, digest, size):
        data = self.files[self.path(digest)]
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Remote recording checksum changed')

    def file_size(self, path, gateway):
        return gateway, len(self.files[path])

    def stream_file_range(self, path, host, *, start, end):
        yield self.files[path][start:end+1]

    def stage(self, digest, size, run_id, gateway):
        assert gateway == 'sky2'
        self.verify(digest, size)
        self.staged.append((digest, run_id, gateway))
