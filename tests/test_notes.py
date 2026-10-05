import itertools
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.changes import CHANNEL
from skynet_app.database import Database
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.metadata_objects import _REMOTE, MAX_BYTES, MetadataObjects
from skynet_app.notes import NoteStore, notes_router, render
from skynet_app.payload_store import _CACHE, PayloadStore
from skynet_app.workspaces import WorkspaceDirectory


@pytest.fixture
def system(tmp_path, monkeypatch):
    def exchange(self, request):
        # Run the real cluster-side object handler locally, without SSH.
        result = subprocess.run([sys.executable, "-c", _REMOTE], input=json.dumps(request), text=True, capture_output=True)
        if result.returncode:
            raise OSError(result.stderr)
        return json.loads(result.stdout)

    monkeypatch.setattr(MetadataObjects, "_exchange", exchange)
    clock = itertools.count(1)
    monkeypatch.setattr("skynet_app.notes.utc_now", lambda: f"2026-10-05T00:00:{next(clock):02d}.000Z")
    return Database(tmp_path / "notes.db")


def workspace(system, email):
    row, _ = WorkspaceDirectory(system).open(email)
    database = system.for_workspace(row["id"])
    database.payload_store = PayloadStore(database)
    return database


def client(services):
    app = FastAPI()
    app.include_router(notes_router(services))
    return TestClient(app)


def add_folder(database, folder_id, name):
    with database.transaction() as connection:
        connection.execute("INSERT INTO note_folders(owner_id,id,name) VALUES (?,?,?)", (database.workspace_id, folder_id, name))


def create(api, title="First", markdown="Body", folder_id=None):
    response = api.post("/api/notes", json={"title": title, "markdown": markdown, "folder_id": folder_id})
    assert response.status_code == 200, response.text
    return response.json()["note"]


def refs(database, table="note_attachments"):
    with database.connection() as connection:
        return connection.execute("SELECT record_id,sha256 FROM metadata_payload_refs WHERE table_name=?", (table,)).fetchall()


@contextmanager
def listening(database):
    connection = database.backend.connect()
    connection.raw.execute("LISTEN " + CHANNEL)
    try:
        yield lambda: [json.loads(item.payload) for item in connection.raw.notifies(timeout=0.1)]
    finally:
        connection.close()


def test_notes_are_workspace_owned_through_the_api_and_the_database_guard(system):
    alice, bob = workspace(system, "alice@example.com"), workspace(system, "bob@example.com")
    services = SimpleNamespace(database=alice)
    with client(services) as api:
        note = create(api, "Private", "Secret")
        assert api.put(f"/api/notes/{note['id']}/attachments/data.json", content=b"{}").status_code == 200
        services.database = bob
        assert api.get("/api/notes").json() == {"notes": [], "folders": []}
        for method, path in [("GET", ""), ("GET", "/download"), ("GET", "/attachments/data.json"), ("DELETE", ""),
                             ("DELETE", "/attachments/data.json")]:
            assert api.request(method, f"/api/notes/{note['id']}{path}").status_code == 404
        update = {"title": "Taken", "markdown": "", "folder_id": None, "expected_updated_at": note["updated_at"]}
        assert api.put(f"/api/notes/{note['id']}", json=update).status_code == 404
        assert api.put(f"/api/notes/{note['id']}/attachments/data.json", content=b"[]").status_code == 404
        services.database = system
        assert api.get("/api/notes").status_code == 401
        assert api.post("/api/notes", json={"title": "x", "markdown": "", "folder_id": None}).status_code == 401
    for statement, values in [
        ("INSERT INTO notes(owner_id,id,title,markdown,created_at,updated_at) VALUES (?,?,?,?,?,?)",
         (alice.workspace_id, "planted", "Planted", "", "t", "t")),
        ("UPDATE notes SET title='Taken' WHERE owner_id=?", (alice.workspace_id,)),
        ("DELETE FROM notes WHERE owner_id=?", (alice.workspace_id,)),
        ("INSERT INTO note_folders(owner_id,id,name) VALUES (?,?,?)", (alice.workspace_id, "f", "Folder")),
    ]:
        with pytest.raises(INTEGRITY_ERRORS), bob.transaction() as connection:
            connection.execute(statement, values)
    with pytest.raises(INTEGRITY_ERRORS), alice.transaction() as connection:
        connection.execute("UPDATE notes SET owner_id=?", (bob.workspace_id,))
    assert NoteStore(alice).read(note["id"])["markdown"] == "Secret"


def test_create_generates_dated_slugs_and_suffixes_collisions(system):
    alice = workspace(system, "alice@example.com")
    with client(SimpleNamespace(database=alice)) as api:
        before = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        first = create(api, "  WARP: MINK vs. WARP — Café 결과  ", "# Body")
        after = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        date = first["id"][:10]
        assert date in {before, after} and first["id"] == f"{date}-warp-mink-vs-warp-cafe"
        assert first["title"] == "WARP: MINK vs. WARP — Café 결과"
        assert first["markdown"] == "# Body" and first["attachments"] == [] and first["folder_id"] is None
        assert first["created_at"] == first["updated_at"]
        assert create(api, "warp mink vs warp cafe")["id"] == f"{date}-warp-mink-vs-warp-cafe-2"
        assert create(api, "WARP MINK vs WARP Cafe")["id"] == f"{date}-warp-mink-vs-warp-cafe-3"
        assert create(api, "실험 계획")["id"] == f"{date}-note"
        assert create(api, "실험 계획")["id"] == f"{date}-note-2"
        long = create(api, "x" * 50 + " " + "y" * 50)["id"]
        assert long == f"{date}-{'x' * 50}-{'y' * 9}" and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", long)
        assert api.post("/api/notes", json={"title": " ", "markdown": "", "folder_id": None}).status_code == 422
        assert api.post("/api/notes", json={"title": "x", "markdown": ""}).status_code == 422
        assert api.post("/api/notes", json={"title": "x", "markdown": "", "folder_id": "nope"}).json()["detail"] == "Choose an existing folder"


def test_listing_orders_notes_by_creation_and_folders_by_name(system):
    alice = workspace(system, "alice@example.com")
    add_folder(alice, "warp", "warp-extension")
    add_folder(alice, "ego", "Egoisim")
    with alice.transaction() as connection:
        for identifier, created, updated, folder in [
            ("old", "2026-09-01T00:00:00.000Z", "2026-09-30T00:00:00.000Z", "ego"),
            ("new", "2026-09-10T00:00:00.000Z", "2026-09-11T00:00:00.000Z", None),
            ("same-a", "2026-09-05T00:00:00.000Z", "2026-09-05T00:00:00.000Z", "warp"),
            ("same-b", "2026-09-05T00:00:00.000Z", "2026-09-05T00:00:00.000Z", None),
        ]:
            connection.execute(
                "INSERT INTO notes(owner_id,id,title,markdown,folder_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (alice.workspace_id, identifier, identifier.title(), "", folder, created, updated),
            )
    with client(SimpleNamespace(database=alice)) as api:
        listing = api.get("/api/notes").json()
    assert [note["id"] for note in listing["notes"]] == ["new", "same-b", "same-a", "old"], "A later edit does not move a note up"
    assert listing["notes"][-1] == {"id": "old", "title": "Old", "folder_id": "ego",
                                    "created_at": "2026-09-01T00:00:00.000Z", "updated_at": "2026-09-30T00:00:00.000Z"}
    assert listing["folders"] == [{"id": "ego", "name": "Egoisim"}, {"id": "warp", "name": "warp-extension"}]


def test_update_validates_and_refuses_stale_edits(system):
    alice = workspace(system, "alice@example.com")
    add_folder(alice, "ego", "Egoisim")
    with client(SimpleNamespace(database=alice)) as api:
        note = create(api, "First", "Body")
        path = f"/api/notes/{note['id']}"
        body = {"title": "First", "markdown": "Body", "folder_id": "ego", "expected_updated_at": note["updated_at"]}
        moved = api.put(path, json=body).json()["note"]
        assert moved["folder_id"] == "ego" and moved["updated_at"] == note["updated_at"], "A move is not an edit"
        edited = api.put(path, json={**body, "title": " Renamed ", "markdown": "New"}).json()["note"]
        assert edited["title"] == "Renamed" and edited["markdown"] == "New" and edited["updated_at"] > note["updated_at"]
        stale = api.put(path, json={**body, "markdown": "Lost"})
        assert stale.status_code == 409 and "changed elsewhere" in stale.json()["detail"]
        current = {**body, "expected_updated_at": edited["updated_at"]}
        for invalid in [{**current, "title": "   "}, {**current, "title": "x" * 241}, {**current, "extra": 1},
                        {**current, "markdown": "x" * (1024 * 1024 + 1)}, {**current, "markdown": "가" * 400000},
                        {**current, "markdown": json.dumps({"$skynet_object_v1": {"sha256": "0" * 64}})}]:
            assert api.put(path, json=invalid).status_code == 422
        unknown = api.put(path, json={**current, "folder_id": "missing"})
        assert unknown.status_code == 422 and unknown.json()["detail"] == "Choose an existing folder"
        assert api.get(path).json()["note"]["markdown"] == "New"
        assert api.put("/api/notes/absent", json=current).status_code == 404


def test_text_fields_refuse_object_references_and_nul(system):
    # Database reads resolve an object reference in any column, so a stored title that is
    # one would break the listing or show another workspace's object body.
    forged = json.dumps({"$skynet_object_v1": {"sha256": "0" * 64, "path": "/x", "size": 1}})
    with client(SimpleNamespace(database=workspace(system, "alice@example.com"))) as api:
        mine = create(api, "Mine")
        for text, message in [({"title": forged}, "Object references"), ({"markdown": forged}, "Object references"),
                              ({"title": "x\x00y"}, "NUL"), ({"markdown": "a\x00b"}, "NUL")]:
            body = {"title": "x", "markdown": "", "folder_id": None, **text}
            for response in [api.post("/api/notes", json=body),
                             api.put(f"/api/notes/{mine['id']}", json={**body, "expected_updated_at": mine["updated_at"]})]:
                assert response.status_code == 422 and message in response.text
        assert [note["title"] for note in api.get("/api/notes").json()["notes"]] == ["Mine"]


def test_read_download_and_delete_release_attachment_references(system):
    alice = workspace(system, "alice@example.com")
    with client(SimpleNamespace(database=alice)) as api:
        note = create(api, "First", "# Other heading\n\nBody")
        keep = create(api, "Keep", "Body")
        for target, name in [(note, "a.svg"), (note, "b.json"), (keep, "a.svg")]:
            assert api.put(f"/api/notes/{target['id']}/attachments/{name}", content=name.encode()).status_code == 200
        read = api.get(f"/api/notes/{note['id']}").json()["note"]
        assert set(read) == {"id", "title", "folder_id", "created_at", "updated_at", "markdown", "html", "attachments"}
        assert read["attachments"] == [
            {"name": "a.svg", "media_type": "image/svg+xml", "size_bytes": 5, "url": f"/api/notes/{note['id']}/attachments/a.svg"},
            {"name": "b.json", "media_type": "application/json", "size_bytes": 6, "url": f"/api/notes/{note['id']}/attachments/b.json"},
        ]
        download = api.get(f"/api/notes/{note['id']}/download")
        assert download.content == b"# Other heading\n\nBody"
        assert download.headers["content-type"] == "text/markdown; charset=utf-8"
        assert download.headers["content-disposition"] == f'attachment; filename="{note["id"]}.md"'
        assert download.headers["cache-control"] == "no-store"
        assert len(refs(alice)) == 3
        assert api.delete(f"/api/notes/{note['id']}").json() == {"deleted": note["id"]}
        assert api.get(f"/api/notes/{note['id']}").status_code == 404
        assert api.delete(f"/api/notes/{note['id']}").status_code == 404
        with alice.connection() as connection:
            remaining = connection.execute("SELECT id,note_id FROM note_attachments").fetchall()
            assert [row["note_id"] for row in remaining] == [keep["id"]]
            assert [row[0] for row in refs(alice)] == [remaining[0]["id"]]
            # The body stays registered until Storage cleanup removes unreferenced objects.
            assert connection.execute("SELECT count(*) FROM metadata_payloads").fetchone()[0] == 2
        assert api.get(f"/api/notes/{keep['id']}/attachments/a.svg").content == b"a.svg"


def test_attachments_upload_serve_replace_and_delete(system):
    alice = workspace(system, "alice@example.com")
    with client(SimpleNamespace(database=alice)) as api:
        note = create(api, "First", "Body")
        base = f"/api/notes/{note['id']}/attachments"
        binary = b"\x89PNG\r\n\x1a\n\x00\xff\xfe" + bytes(range(256))
        saved = api.put(f"{base}/plot.png", content=binary, headers={"Content-Type": "application/octet-stream"})
        assert saved.json() == {"attachment": {"name": "plot.png", "media_type": "image/png", "size_bytes": len(binary),
                                               "url": f"{base}/plot.png"}}
        _CACHE.clear()
        served = api.get(f"{base}/plot.png")
        assert served.content == binary and served.headers["content-type"] == "image/png"
        assert served.headers["x-content-type-options"] == "nosniff"
        assert served.headers["content-security-policy"] == "sandbox"
        page = b"<script>alert(1)</script>"
        assert api.put(f"{base}/report.html", content=page, headers={"Content-Type": "text/html"}).status_code == 200
        served = api.get(f"{base}/report.html")
        assert served.content == page and served.headers["content-type"] == "text/html; charset=utf-8"
        assert served.headers["content-security-policy"] == "sandbox"
        uploaded = api.get(f"/api/notes/{note['id']}").json()["note"]["updated_at"]
        assert uploaded > note["updated_at"]
        assert api.put(f"{base}/plot.png", content=binary).status_code == 200
        assert api.get(f"/api/notes/{note['id']}").json()["note"]["updated_at"] == uploaded, "Identical bytes change nothing"
        assert api.put(f"{base}/plot.png", content=b"replaced").json()["attachment"]["size_bytes"] == 8
        assert api.get(f"{base}/plot.png").content == b"replaced"
        replaced = api.get(f"/api/notes/{note['id']}").json()["note"]
        assert replaced["updated_at"] > uploaded and [item["name"] for item in replaced["attachments"]] == ["plot.png", "report.html"]
        assert len(refs(alice)) == 2
        for name in ["run.exe", "noext", ".hidden.svg", "spaced%20name.svg", "x" * 129 + ".svg"]:
            assert api.put(f"{base}/{name}", content=b"x").status_code == 422
        assert api.put(f"{base}/huge.svg", content=b"x" * (MAX_BYTES + 1)).status_code == 413
        assert api.put("/api/notes/absent/attachments/x.svg", content=b"x").status_code == 404
        assert api.get(f"{base}/missing.svg").status_code == 404
        assert api.delete(f"{base}/plot.png").json() == {"deleted": "plot.png"}
        assert api.get(f"{base}/plot.png").status_code == 404
        assert api.delete(f"{base}/plot.png").status_code == 404
        assert api.get(f"/api/notes/{note['id']}").json()["note"]["updated_at"] > replaced["updated_at"]
        assert len(refs(alice)) == 1
        alice.payload_store = None
        unavailable = api.put(f"{base}/later.svg", content=b"x")
        assert unavailable.status_code == 503 and "object store" in unavailable.json()["detail"]


def test_renderer_links_figures_headings_and_tables(system):
    attachments = {"plot.svg": "image/svg+xml", "metrics.json": "application/json"}
    html = render("first", "First", (
        "# First\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1))\n\n"
        "[second](second.md) [missing](missing.md) [nested](dir/second.md) [metrics](metrics.json) "
        "[folder](/?experiment_view=notes&note_folder=ego#experiments) [static](/static/app.js) "
        "[file](/Users/me/notes/second.md) [official](https://example.com) [old](file:///etc/passwd)\n\n"
        "![Plot](plot.svg) ![tracker](https://example.com/tracker.png) ![data](metrics.json) ![](absent.png)\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |"
    ), {"first", "second"}, attachments)
    assert "<h1>" not in html and "<script>" not in html and 'href="javascript:' not in html
    assert '<a href="/?experiment_view=notes&amp;note=second#experiments">second</a>' in html
    assert '<span title="missing.md">missing</span>' in html and '<span title="dir/second.md">nested</span>' in html
    assert '<a href="/api/notes/first/attachments/metrics.json" target="_blank" rel="noopener noreferrer">metrics</a>' in html
    assert '<a href="/?experiment_view=notes&amp;note_folder=ego#experiments">folder</a>' in html
    assert '<span title="/static/app.js">static</span>' in html
    assert '<span title="/Users/me/notes/second.md">file</span>' in html
    assert '<a href="https://example.com" target="_blank" rel="noopener noreferrer">official</a>' in html
    assert ('<a class="note-figure" href="/api/notes/first/attachments/plot.svg" target="_blank" rel="noopener noreferrer">'
            '<img src="/api/notes/first/attachments/plot.svg" alt="Plot" loading="eager" decoding="async"></a>' in html)
    assert html.count("<img") == 1 and " tracker data Image</p>" in html
    assert '<div class="table-scroll identity-table"><table>' in html and "</table></div>" in html
    assert "<h1>Different</h1>" in render("first", "First", "# Different\n\nBody", set(), {})
    assert "<h1>" not in render("first", "First", "Intro\n\n#  First \n\nBody", set(), {})
    assert '<span title="plot.svg">other</span>' in render("second", "Second", "[other](plot.svg)", set(), {})
    linked = render("first", "First", "[![Plot](plot.svg)](https://example.com) [![Plot](plot.svg)](missing.md)", set(), attachments)
    assert ('<a href="https://example.com" target="_blank" rel="noopener noreferrer"><img src="/api/notes/first/attachments/plot.svg"'
            ' alt="Plot" loading="eager" decoding="async"></a>' in linked), "Anchors never nest"
    assert '<span title="missing.md"><a class="note-figure" href="/api/notes/first/attachments/plot.svg" target="_blank"' in linked


def test_rendered_notes_resolve_links_and_attachments_from_the_workspace(system):
    alice, bob = workspace(system, "alice@example.com"), workspace(system, "bob@example.com")
    with client(SimpleNamespace(database=bob)) as api:
        foreign = create(api, "Foreign")
    with client(SimpleNamespace(database=alice)) as api:
        second = create(api, "Second")
        note = create(api, "First", f"# First\n\n[s]({second['id']}.md) [f]({foreign['id']}.md)\n\n![Plot](plot.svg)")
        assert "<img" not in api.get(f"/api/notes/{note['id']}").json()["note"]["html"]
        api.put(f"/api/notes/{note['id']}/attachments/plot.svg", content=b"<svg/>")
        html = api.get(f"/api/notes/{note['id']}").json()["note"]["html"]
    assert f'<a href="/?experiment_view=notes&amp;note={second["id"]}#experiments">s</a>' in html
    assert f'<span title="{foreign["id"]}.md">f</span>' in html
    assert f'<img src="/api/notes/{note["id"]}/attachments/plot.svg"' in html and "<h1>" not in html


def test_math_is_left_as_escaped_tex_for_katex():
    body = ('# First\n\nInline $a_i<b$, not money: $5 and $6, code `$x$`.\n\n$$\n\\frac{1}{k}\\sum_j x_j\n$$\n\n'
            '| Metric | Formula |\n|---|---|\n| MI | $H(A\\|S)$ |')
    html = render("first", "First", body, set(), {})
    assert '<span class="math math-inline">a_i&lt;b</span>' in html
    assert '$5 and $6' in html and '<code>$x$</code>' in html
    assert '<div class="math math-display">\\frac{1}{k}\\sum_j x_j</div>' in html
    assert '<span class="math math-inline">H(A|S)</span>' in html


def test_folder_controls_stay_removed(system):
    alice = workspace(system, "alice@example.com")
    add_folder(alice, "ego", "Egoisim")
    with client(SimpleNamespace(database=alice)) as api:
        note = create(api, "First", folder_id="ego")
        for method, path, body in [
            ("POST", "/api/notes/folders", {"name": "New"}),
            ("PATCH", "/api/notes/folders/ego", {"name": "Renamed"}),
            ("DELETE", "/api/notes/folders/ego", None),
            ("POST", "/api/notes/move", {"note_ids": [note["id"]], "folder_id": None}),
        ]:
            assert api.request(method, path, json=body).status_code in {404, 405}
        assert api.get("/api/notes").json()["folders"] == [{"id": "ego", "name": "Egoisim"}]


def test_notes_topic_is_notified_for_the_owner_only_on_real_changes(system):
    alice = workspace(system, "alice@example.com")
    with listening(system) as notifications, client(SimpleNamespace(database=alice)) as api:
        expected = [{"v": 1, "scope": alice.workspace_id, "topics": ["notes"]}]
        add_folder(alice, "ego", "Egoisim")
        assert notifications() == expected
        note = create(api, "First")
        assert notifications() == expected
        body = {"title": "First", "markdown": "Body", "folder_id": None, "expected_updated_at": note["updated_at"]}
        assert api.put(f"/api/notes/{note['id']}", json=body).status_code == 200
        assert notifications() == []
        api.put(f"/api/notes/{note['id']}/attachments/a.svg", content=b"<svg/>")
        assert notifications() == expected
        api.put(f"/api/notes/{note['id']}/attachments/a.svg", content=b"<svg/>")
        assert notifications() == []
        api.delete(f"/api/notes/{note['id']}")
        assert notifications() == expected
