"""Pangram AI-detection client (async task API).

Docs: https://docs.pangram.com/api-reference/ai-detection
Key: PANGRAM_API_KEY, read from the environment or the project's .env file
(re-read on every call, so you can add the key without restarting).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values

BASE = "https://text.external-api.pangram.com"
MODEL = "pangram-4"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
MIN_WORDS = 20  # don't spend credits on fragments

_cache: dict[str, dict] = {}


class PangramError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("PANGRAM_API_KEY") or dotenv_values(ENV_FILE).get("PANGRAM_API_KEY")


def _summary(r: dict) -> dict:
    keep = (
        "headline", "prediction", "prediction_short", "fraction_ai", "fraction_ai_assisted",
        "fraction_human", "num_ai_segments", "num_ai_assisted_segments", "num_human_segments", "version",
    )
    out = {k: r.get(k) for k in keep}
    out["windows"] = [
        {k: w.get(k) for k in ("label", "ai_assistance_score", "confidence", "start_index", "end_index")}
        for w in r.get("windows") or []
    ]
    return out


async def detect(text: str, timeout: float = 60.0) -> dict:
    key = api_key()
    if not key:
        raise PangramError("no PANGRAM_API_KEY in environment or .env")
    if len(text.split()) < MIN_WORDS:
        raise PangramError(f"need at least {MIN_WORDS} words")
    h = hashlib.sha256(text.encode()).hexdigest()
    if h in _cache:
        return _cache[h]

    headers = {"x-api-key": key}
    async with httpx.AsyncClient(base_url=BASE, headers=headers, timeout=20) as c:
        r = await c.post("/task", json={"text": text, "model": MODEL, "public_dashboard_link": False})
        if r.status_code != 200:
            raise PangramError(f"POST /task {r.status_code}: {r.text[:200]}")
        task_id = r.json()["task_id"]
        loop = asyncio.get_running_loop()
        deadline, delay = loop.time() + timeout, 0.4
        while True:
            r = await c.get(f"/task/{task_id}")
            if r.status_code != 200:
                raise PangramError(f"GET /task {r.status_code}: {r.text[:200]}")
            j = r.json()
            if j.get("stage") == "STAGE_SUCCESS":
                break
            if j.get("stage") == "STAGE_FAILED":
                raise PangramError(j.get("error") or "task failed")
            if loop.time() > deadline:
                raise PangramError("timed out waiting for Pangram")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 2.0)

    out = _summary(j)
    _cache[h] = out
    return out
