"""Reducto PDF parsing -> paragraph-sized chunks for the editor's linear import chain.

Docs: https://docs.reducto.ai (POST /upload, POST /parse)
Key: REDUCTO_API_KEY, read from the environment or the project's .env (re-read per call).

We ask Reducto for its layout blocks (Title / Section Header / Text / List Item ...),
drop page furniture (headers, footers, page numbers, figures, tables), and group the
rest into paragraph-sized chunks: a heading travels with the paragraph under it, and
very short paragraphs merge into the next one so every chunk is worth a decision node.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

import httpx
from dotenv import dotenv_values

BASE = "https://platform.reducto.ai"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

SKIP = {"Header", "Footer", "Page Number", "Figure", "Table", "Comment", "Signature"}
HEADINGS = {"Title", "Section Header"}
MIN_CHUNK_WORDS = 25      # shorter paragraphs merge into the next chunk
MAX_WORDS = 8000          # cap the import (Pangram credits, editor size)


class ReductoError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("REDUCTO_API_KEY") or dotenv_values(ENV_FILE).get("REDUCTO_API_KEY")


def _clean(s: str) -> str:
    s = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", "", s))   # Reducto marks emphasis with HTML tags
    s = re.sub(r"^#+\s*", "", s.strip())            # markdown heading marks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)                 # PDF line wraps inside a block
    return s.strip()


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """Group Reducto blocks into paragraph chunks: [{text, page, words}]."""
    chunks: list[dict] = []
    heads: list[str] = []
    pending: dict | None = None   # a short paragraph waiting to merge forward
    page = 1

    def emit(text: str, page: int):
        nonlocal pending
        if pending:
            text, page = pending["text"] + "\n\n" + text, pending["page"]
            pending = None
        c = {"text": text, "page": page, "words": len(text.split())}
        if c["words"] < MIN_CHUNK_WORDS:
            pending = c
        else:
            chunks.append(c)

    for b in blocks:
        kind = b.get("type") or "Text"
        if kind in SKIP:
            continue
        text = _clean(b.get("content") or "")
        if not text:
            continue
        page = (b.get("bbox") or {}).get("original_page") or (b.get("bbox") or {}).get("page") or 1
        if kind in HEADINGS:
            heads.append(text)
            continue
        if heads:
            text = "\n".join(heads) + "\n" + text
            heads = []
        emit(text, page)
    if heads:
        emit("\n".join(heads), page)
    if pending:
        if chunks:   # trailing short bit joins the last chunk
            last = chunks[-1]
            last["text"] += "\n\n" + pending["text"]
            last["words"] = len(last["text"].split())
        else:
            chunks.append(pending)
    return chunks


async def parse_pdf(data: bytes, filename: str = "upload.pdf", timeout: float = 180.0) -> dict:
    key = api_key()
    if not key:
        raise ReductoError("no REDUCTO_API_KEY in environment or .env")
    if not data.startswith(b"%PDF"):
        raise ReductoError("that file is not a PDF")
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=BASE, headers=headers, timeout=timeout) as c:
        r = await c.post("/upload", files={"file": (filename, data, "application/pdf")})
        if r.status_code != 200:
            raise ReductoError(f"upload {r.status_code}: {r.text[:200]}")
        file_id = r.json()["file_id"]
        r = await c.post("/parse", json={"input": file_id})
        if r.status_code != 200:
            raise ReductoError(f"parse {r.status_code}: {r.text[:200]}")
        j = r.json()
        res = j.get("result") or {}
        if res.get("type") == "url":            # large documents come back by reference
            res = (await c.get(res["url"])).json()
            res = res.get("result", res) if isinstance(res, dict) else {"chunks": res}
    blocks = [b for ch in res.get("chunks") or [] for b in ch.get("blocks") or []]
    chunks = chunk_blocks(blocks)
    kept, total = [], 0
    for ch in chunks:
        if total + ch["words"] > MAX_WORDS:
            break
        kept.append(ch)
        total += ch["words"]
    usage = j.get("usage") or {}
    return {
        "chunks": kept,
        "truncated": len(kept) < len(chunks),
        "words": total,
        "pages": usage.get("num_pages"),
        "blocks": len(blocks),
        "job_id": j.get("job_id"),
    }
