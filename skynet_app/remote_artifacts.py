"""Serve verified remote artifacts without a persistent operator-machine cache."""

from dataclasses import dataclass
import re
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import StreamingResponse


@dataclass(frozen=True)
class RemoteArtifact:
    transport: object
    gateway: str
    path: str
    max_bytes: int | None = None

    def location(self):
        host, size = self.transport.file_size(self.path, self.gateway)
        if size < 1 or (self.max_bytes is not None and size > self.max_bytes):
            raise ValueError("Remote artifact is empty or exceeds its size limit")
        return host, size

    def read_bytes(self):
        if self.max_bytes is None:
            raise ValueError("Reading an artifact into memory requires a size limit")
        host, size = self.location()
        data = bytearray()
        for block in self.transport.stream_file_range(self.path, host, start=0, end=size - 1):
            data.extend(block)
            if len(data) > size:
                raise ValueError("Remote artifact changed while reading")
        if len(data) != size:
            raise ValueError("Remote artifact transfer was incomplete")
        return bytes(data)

    def read_text(self):
        return self.read_bytes().decode("utf-8")

    def response(self, request, media_type="application/octet-stream", filename=None):
        host, size = self.location()
        start, end, partial = 0, size - 1, False
        value = request.headers.get("range")
        if value:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
            if not match or not any(match.groups()):
                raise HTTPException(416, "Unsupported byte range", headers={"Content-Range": f"bytes */{size}"})
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                valid = start < size and end >= start
            else:
                suffix = int(match[2])
                start = max(0, size - suffix)
                valid = suffix > 0
            if not valid:
                raise HTTPException(416, "Byte range is outside the file", headers={"Content-Range": f"bytes */{size}"})
            partial = True
        headers = {
            "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1),
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Content-Disposition": "inline" if filename is None else "attachment; filename*=UTF-8''" + quote(filename, safe=""),
        }
        if partial:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return StreamingResponse(
            self.transport.stream_file_range(self.path, host, start=start, end=end),
            status_code=206 if partial else 200, media_type=media_type, headers=headers,
        )
