"""Document store for the editor: one JSON file per document under documents/.

A document is {id, title, created, updated, html, text, tree, pangram}. `html` keeps
the authorship spans, `tree` is the full decision tree, `pangram` the last result.
Writes are atomic (tmp file + rename) so a crash never leaves half a document.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path

ROOT = Path(os.environ.get("COWRITE_DOCS") or Path(__file__).resolve().parent.parent / "documents")
ID_RE = re.compile(r"^[a-z0-9]{8,32}$")
FIELDS = ("title", "html", "text", "tree", "pangram")
MAX_BYTES = 20_000_000


class DocError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _path(doc_id: str) -> Path:
    if not ID_RE.match(doc_id or ""):
        raise DocError("bad document id")
    return ROOT / f"{doc_id}.json"


def _write(doc: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False)
    if len(data) > MAX_BYTES:
        raise DocError("document too large", 413)
    p = _path(doc["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(p)


def _summary(doc: dict) -> dict:
    text = doc.get("text") or ""
    return {
        "id": doc["id"], "title": doc.get("title") or "Untitled",
        "created": doc.get("created"), "updated": doc.get("updated"),
        "words": len(text.split()), "preview": " ".join(text.split())[:120],
    }


def list_docs() -> list[dict]:
    if not ROOT.exists():
        return []
    out = []
    for p in ROOT.glob("*.json"):
        try:
            out.append(_summary(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue   # skip unreadable files rather than failing the whole list
    return sorted(out, key=lambda d: d.get("updated") or 0, reverse=True)


def create(body: dict) -> dict:
    now = time.time()
    doc = {"id": secrets.token_hex(8), "title": "Untitled", "created": now, "updated": now,
           "html": "", "text": "", "tree": None, "pangram": None}
    doc.update({k: body[k] for k in FIELDS if k in body})
    doc["title"] = (doc.get("title") or "Untitled").strip()[:200] or "Untitled"
    _write(doc)
    return doc


def read(doc_id: str) -> dict:
    p = _path(doc_id)
    if not p.exists():
        raise DocError("no such document", 404)
    return json.loads(p.read_text(encoding="utf-8"))


def update(doc_id: str, body: dict) -> dict:
    doc = read(doc_id)
    doc.update({k: body[k] for k in FIELDS if k in body})
    doc["title"] = (doc.get("title") or "Untitled").strip()[:200] or "Untitled"
    doc["updated"] = time.time()
    _write(doc)
    return _summary(doc)


def delete(doc_id: str) -> None:
    p = _path(doc_id)
    if not p.exists():
        raise DocError("no such document", 404)
    p.unlink()
