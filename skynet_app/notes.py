"""Workspace Markdown notes, folders and attachments in the central database."""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import re
import unicodedata
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator
from starlette.concurrency import run_in_threadpool

from .database import new_id, utc_now
from .metadata_objects import MAX_BYTES
from .payload_store import reference

MAX_MARKDOWN_BYTES = 1024 * 1024
ATTACHMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MEDIA_TYPES = {
    "svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "webp": "image/webp", "gif": "image/gif", "json": "application/json", "html": "text/html",
    "csv": "text/csv", "txt": "text/plain", "md": "text/markdown",
}
OBJECT_STORE_REQUIRED = "Note attachments need the central object store. Configure object_store_root for this app."


class NoteConflict(Exception):
    """A write based on a version of the note that is no longer current."""


class ObjectStoreUnavailable(Exception):
    pass


def storable(value: str) -> str:
    """Text for a notes column: PostgreSQL refuses NUL, and database reads resolve object references."""
    if "\x00" in value:
        raise ValueError("Notes cannot contain NUL characters")
    if reference(value):
        raise ValueError("Object references cannot be supplied as note content")
    return value


class NoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
    markdown: str
    folder_id: str | None

    @field_validator("title", "markdown")
    @classmethod
    def _storable(cls, value: str) -> str:
        return storable(value)

    @field_validator("markdown")
    @classmethod
    def _markdown(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_MARKDOWN_BYTES:
            raise ValueError("A note can contain at most 1 MiB of Markdown")
        return value


class NoteUpdate(NoteRequest):
    expected_updated_at: str


def attachment_type(name: str) -> str:
    if not ATTACHMENT_NAME.fullmatch(name):
        raise ValueError("Attachment names use letters, numbers, dots, underscores or hyphens and start with a letter or number")
    media_type = MEDIA_TYPES.get(name.rsplit(".", 1)[-1].lower() if "." in name else "")
    if media_type is None:
        raise ValueError("Attach " + ", ".join(sorted(MEDIA_TYPES)) + " files")
    return media_type


def attachment_url(note_id: str, name: str) -> str:
    return f"/api/notes/{note_id}/attachments/{name}"


def _escaped(src: str, pos: int) -> bool:
    return (len(src[:pos]) - len(src[:pos].rstrip("\\"))) % 2 == 1


def _math_inline(state, silent: bool) -> bool:
    """`$…$` and `$$…$$`; a single `$` follows Pandoc's rules so `$5 and $6` stays text, and math never runs into code."""
    src, start = state.src, state.pos
    if src[start] != "$":
        return False
    mark = "$$" if src.startswith("$$", start) else "$"
    close = start + len(mark)
    while (close := src.find(mark, close, state.posMax)) != -1 and _escaped(src, close):
        close += 1
    if close == -1:
        return False
    tex = src[start + len(mark):close]
    if not tex.strip() or "`" in tex or (mark == "$" and (tex[0].isspace() or tex[-1].isspace() or src[close + 1:close + 2].isdigit())):
        return False
    if not silent:
        token = state.push("math_inline", "math", 0)
        token.content, token.markup = tex.strip(), mark
    state.pos = close + len(mark)
    return True


def _math_block(state, start: int, end: int, silent: bool) -> bool:
    """Display math on its own lines: `$$ … $$`, or `$$` … `$$` across lines."""
    if state.sCount[start] - state.blkIndent >= 4:
        return False
    line = lambda n: state.src[state.bMarks[n] + state.tShift[n]:state.eMarks[n]].strip()
    first = line(start)
    if not first.startswith("$$") or ("$$" in first[2:] and not first.endswith("$$")):
        return False
    body, last = [first[2:]], start
    while not body[-1].endswith("$$"):
        last += 1
        if last >= end:
            return False
        body.append(line(last))
    tex = "\n".join(body)[:-2].strip()
    if not tex:
        return False
    if silent:
        return True
    token = state.push("math_block", "math", 0)
    token.content, token.markup, token.map = tex, "$$", [start, last + 1]
    state.line = last + 1
    return True


def _render_math(self, tokens, idx, options, env) -> str:
    # The browser typesets the escaped TeX with KaTeX; without it the source stays readable.
    token = tokens[idx]
    tag, kind = ("div" if token.block else "span"), ("display" if token.markup == "$$" else "inline")
    return f'<{tag} class="math math-{kind}">{escapeHtml(token.content)}</{tag}>' + ("\n" if token.block else "")


def _render_figure(self, tokens, idx, options, env) -> str:
    token = tokens[idx]
    url = escapeHtml(token.attrGet("src"))
    alt = escapeHtml(self.renderInlineAsText(token.children or [], options, env))
    image = f'<img src="{url}" alt="{alt}" loading="eager" decoding="async">'
    # Inside a link the image is the link's content; anchors cannot nest.
    return image if token.meta.get("linked") else f'<a class="note-figure" href="{url}" target="_blank" rel="noopener noreferrer">{image}</a>'


def markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    parser.block.ruler.before("fence", "math_block", _math_block, {"alt": ["paragraph", "reference", "blockquote", "list"]})
    parser.inline.ruler.before("escape", "math_inline", _math_inline)
    parser.add_render_rule("math_inline", _render_math)
    parser.add_render_rule("math_block", _render_math)
    parser.add_render_rule("image", _render_figure)
    return parser


def render(note_id: str, title: str, markdown: str, note_ids: set[str], attachments: dict[str, str]) -> str:
    """HTML without raw markup; only in-app, attachment and web links stay clickable."""
    parser = markdown_parser()
    tokens = parser.parse(markdown)
    for block in tokens:
        spans = []
        for token in block.children or []:
            if token.type == "image":
                name = token.attrGet("src") or ""
                if attachments.get(name, "").startswith("image/"):
                    token.attrSet("src", attachment_url(note_id, name))
                    token.meta["linked"] = bool(spans) and not spans[-1]
                    continue
                # Notes never load remote images: keep the alternative text.
                alt = parser.renderer.renderInlineAsText(token.children or [], parser.options, {})
                token.type, token.tag, token.attrs, token.children = "text", "", {}, None
                token.content = alt or "Image"
            if token.type == "link_open":
                href = token.attrGet("href") or ""
                url = urlsplit(href)
                if url.scheme in {"http", "https"}:
                    token.attrSet("target", "_blank")
                    token.attrSet("rel", "noopener noreferrer")
                elif url.scheme or url.netloc:
                    token.tag, token.attrs = "span", {"title": href}
                elif url.path in attachments:
                    # Attachments open beside the app, like web links.
                    token.attrSet("href", attachment_url(note_id, url.path))
                    token.attrSet("target", "_blank")
                    token.attrSet("rel", "noopener noreferrer")
                elif url.path.endswith(".md") and url.path[:-3] in note_ids:
                    token.attrSet("href", f"/?experiment_view=notes&note={url.path[:-3]}#experiments")
                elif url.path != "/":
                    token.tag, token.attrs = "span", {"title": href}
                spans.append(token.tag == "span")
            elif token.type == "link_close" and spans and spans.pop():
                token.tag = "span"
    heading = next((index for index, token in enumerate(tokens)
                    if token.type == "heading_open" and token.tag == "h1" and token.level == 0), None)
    if heading is not None and tokens[heading + 1].content.strip() == title.strip():
        tokens = tokens[:heading] + tokens[heading + 3:]
    rendered = parser.renderer.render(tokens, parser.options, {})
    return rendered.replace("<table>", '<div class="table-scroll identity-table"><table>').replace("</table>", "</table></div>")


def _slug(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title).strip("-")[:60].strip("-") or "note"


class NoteStore:
    def __init__(self, database):
        if database.workspace_id is None:
            raise ValueError("A workspace is required for notes")
        self.database = database
        self.owner = database.workspace_id

    def _note(self, connection, identifier: str, columns="id,title,folder_id,created_at,updated_at"):
        row = connection.execute(f"SELECT {columns} FROM notes WHERE owner_id=? AND id=?", (self.owner, identifier)).fetchone()
        if row is None:
            raise KeyError("Note not found")
        return dict(row)

    def _check_folder(self, connection, folder_id: str | None) -> None:
        if folder_id is not None and connection.execute(
            "SELECT 1 FROM note_folders WHERE owner_id=? AND id=?", (self.owner, folder_id)
        ).fetchone() is None:
            raise ValueError("Choose an existing folder")

    def _touch(self, connection, identifier: str) -> None:
        connection.execute("UPDATE notes SET updated_at=? WHERE owner_id=? AND id=?", (utc_now(), self.owner, identifier))

    def _objects(self):
        if self.database.payload_store is None:
            raise ObjectStoreUnavailable(OBJECT_STORE_REQUIRED)
        return self.database.payload_store

    def listing(self) -> dict:
        with self.database.read_snapshot() as connection:
            notes = [dict(row) for row in connection.execute(
                "SELECT id,title,folder_id,created_at,updated_at FROM notes WHERE owner_id=?", (self.owner,)
            ).fetchall()]
            folders = [dict(row) for row in connection.execute(
                "SELECT id,name FROM note_folders WHERE owner_id=?", (self.owner,)
            ).fetchall()]
        return {
            "notes": sorted(notes, key=lambda note: (note["created_at"], note["id"]), reverse=True),
            "folders": sorted(folders, key=lambda folder: (folder["name"].casefold(), folder["id"])),
        }

    def read(self, identifier: str) -> dict:
        with self.database.read_snapshot() as connection:
            note = self._note(connection, identifier, "id,title,folder_id,created_at,updated_at,markdown")
            attachments = [dict(row) for row in connection.execute(
                "SELECT name,media_type,size_bytes FROM note_attachments WHERE owner_id=? AND note_id=? ORDER BY name",
                (self.owner, identifier),
            ).fetchall()]
            note_ids = {row[0] for row in connection.execute("SELECT id FROM notes WHERE owner_id=?", (self.owner,)).fetchall()}
        types = {item["name"]: item["media_type"] for item in attachments}
        return {
            **note,
            "html": render(identifier, note["title"], note["markdown"], note_ids, types),
            "attachments": [{**item, "url": attachment_url(identifier, item["name"])} for item in attachments],
        }

    def markdown(self, identifier: str) -> str:
        with self.database.connection() as connection:
            return self._note(connection, identifier, "markdown")["markdown"]

    def create(self, title: str, markdown: str, folder_id: str | None) -> dict:
        base = datetime.now(timezone.utc).strftime("%Y-%m-%d-") + _slug(title)
        with self.database.transaction() as connection:
            self._check_folder(connection, folder_id)
            taken = {row[0] for row in connection.execute(
                "SELECT id FROM notes WHERE owner_id=? AND (id=? OR id LIKE ?)", (self.owner, base, base + "-%")
            ).fetchall()}
            identifier = next(candidate for candidate in itertools.chain(
                [base], (f"{base}-{number}" for number in itertools.count(2))
            ) if candidate not in taken)
            now = utc_now()
            connection.execute(
                "INSERT INTO notes(owner_id,id,title,markdown,folder_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (self.owner, identifier, title, markdown, folder_id, now, now),
            )
        return self.read(identifier)

    def update(self, identifier: str, title: str, markdown: str, folder_id: str | None, expected_updated_at: str) -> dict:
        with self.database.transaction() as connection:
            current = self._note(connection, identifier, "title,markdown,updated_at")
            if current["updated_at"] != expected_updated_at:
                raise NoteConflict("This note changed elsewhere. Reload it and try again.")
            self._check_folder(connection, folder_id)
            changed = (current["title"], current["markdown"]) != (title, markdown)
            connection.execute(
                "UPDATE notes SET title=?,markdown=?,folder_id=?,updated_at=? WHERE owner_id=? AND id=?",
                (title, markdown, folder_id, utc_now() if changed else current["updated_at"], self.owner, identifier),
            )
        return self.read(identifier)

    def delete(self, identifier: str) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """DELETE FROM metadata_payload_refs WHERE table_name='note_attachments' AND record_id IN
                   (SELECT id FROM note_attachments WHERE owner_id=? AND note_id=?)""",
                (self.owner, identifier),
            )
            if connection.execute("DELETE FROM notes WHERE owner_id=? AND id=?", (self.owner, identifier)).rowcount != 1:
                raise KeyError("Note not found")

    def attachment(self, identifier: str, name: str) -> tuple[bytes, str]:
        with self.database.connection() as connection:
            row = connection.execute(
                """SELECT a.media_type,a.sha256,p.path,p.size_bytes FROM note_attachments a
                   JOIN metadata_payloads p ON p.sha256=a.sha256 WHERE a.owner_id=? AND a.note_id=? AND a.name=?""",
                (self.owner, identifier, name),
            ).fetchone()
        if row is None:
            raise KeyError("Attachment not found")
        content = self._objects().read_bytes({"sha256": row["sha256"], "path": row["path"], "size": row["size_bytes"]})
        return content, row["media_type"]

    def save_attachment(self, identifier: str, name: str, content: bytes) -> dict:
        media_type = attachment_type(name)
        store = self._objects()
        with self.database.connection() as connection:
            self._note(connection, identifier)
        # The body is uploaded before the SQL write lock is taken; the row reference
        # recorded with the attachment keeps it from Storage cleanup.
        with self.database.transaction(payloads=(content,)) as connection:
            self._note(connection, identifier)
            current = connection.execute(
                "SELECT id,sha256 FROM note_attachments WHERE owner_id=? AND note_id=? AND name=?",
                (self.owner, identifier, name),
            ).fetchone()
            attachment_id = current["id"] if current else new_id()
            ref = store.put_bytes(connection, "note_attachments", attachment_id, "content", content)
            if current is None or current["sha256"] != ref["sha256"]:
                connection.execute(
                    """INSERT INTO note_attachments(id,owner_id,note_id,name,media_type,size_bytes,sha256,created_at)
                       VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                       media_type=excluded.media_type,size_bytes=excluded.size_bytes,sha256=excluded.sha256""",
                    (attachment_id, self.owner, identifier, name, media_type, len(content), ref["sha256"], utc_now()),
                )
                self._touch(connection, identifier)
        return {"name": name, "media_type": media_type, "size_bytes": len(content), "url": attachment_url(identifier, name)}

    def delete_attachment(self, identifier: str, name: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute(
                "DELETE FROM note_attachments WHERE owner_id=? AND note_id=? AND name=? RETURNING id",
                (self.owner, identifier, name),
            ).fetchone()
            if row is None:
                raise KeyError("Attachment not found")
            connection.execute(
                "DELETE FROM metadata_payload_refs WHERE table_name='note_attachments' AND record_id=?", (row["id"],)
            )
            self._touch(connection, identifier)


def notes_router(services):
    router = APIRouter(prefix="/api/notes", tags=["Notes"])

    def store():
        database = services.database
        if database.workspace_id is None:
            raise HTTPException(401, "Select a workspace")
        return NoteStore(database)

    def checked(operation, *args):
        try:
            return operation(*args)
        except KeyError as error:
            raise HTTPException(404, error.args[0]) from None
        except NoteConflict as error:
            raise HTTPException(409, str(error)) from error
        except ObjectStoreUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("")
    def list_notes():
        return store().listing()

    @router.post("")
    def create_note(request: NoteRequest):
        return {"note": checked(store().create, request.title, request.markdown, request.folder_id)}

    @router.get("/{note_id}")
    def read_note(note_id: str):
        return {"note": checked(store().read, note_id)}

    @router.put("/{note_id}")
    def update_note(note_id: str, request: NoteUpdate):
        return {"note": checked(
            store().update, note_id, request.title, request.markdown, request.folder_id, request.expected_updated_at,
        )}

    @router.delete("/{note_id}")
    def delete_note(note_id: str):
        checked(store().delete, note_id)
        return {"deleted": note_id}

    @router.get("/{note_id}/download")
    def download_note(note_id: str):
        markdown = checked(store().markdown, note_id)
        return Response(markdown, media_type="text/markdown", headers={
            "Content-Disposition": f'attachment; filename="{note_id}.md"', "Cache-Control": "no-store",
        })

    @router.get("/{note_id}/attachments/{name}")
    def read_attachment(note_id: str, name: str):
        content, media_type = checked(store().attachment, note_id, name)
        # Attachments are served from the app origin: never sniff or run their markup.
        return Response(content, media_type=media_type, headers={
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox",
        })

    @router.put("/{note_id}/attachments/{name}")
    async def save_attachment(note_id: str, name: str, request: Request):
        notes = await run_in_threadpool(store)
        content = bytearray()
        async for chunk in request.stream():
            content += chunk
            if len(content) > MAX_BYTES:
                raise HTTPException(413, f"An attachment can be at most {MAX_BYTES // (1024 * 1024)} MiB")
        return {"attachment": await run_in_threadpool(checked, notes.save_attachment, note_id, name, bytes(content))}

    @router.delete("/{note_id}/attachments/{name}")
    def delete_attachment(note_id: str, name: str):
        checked(store().delete_attachment, note_id, name)
        return {"deleted": name}

    return router
