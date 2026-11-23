"""Web co-writing editor backed by the steerable LFM3 engine.

Run:  uv run cosign-web   ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import documents, pangram, reducto
from .engine import MODEL_ID, Engine

DIST = Path(__file__).parent / "web" / "dist"   # built by `pnpm build` in frontend/
app = FastAPI()
engine: Engine | None = None
lock = asyncio.Lock()  # one engine, one request at a time




class DetectReq(BaseModel):
    text: str


@app.post("/api/detect")
async def detect(req: DetectReq):
    try:
        return await pangram.detect(req.text)
    except pangram.PangramError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # network etc.
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.post("/api/import")
async def import_pdf(request: Request):
    """Raw PDF body -> Reducto parse -> paragraph chunks for the linear import chain."""
    data = await request.body()
    name = request.headers.get("x-filename") or "upload.pdf"
    try:
        return await reducto.parse_pdf(data, name)
    except reducto.ReductoError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # network etc.
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


# ---------------------------------------------------------------- documents (CRUD)
def _doc_call(fn, *a):
    try:
        return fn(*a)
    except documents.DocError as e:
        return JSONResponse({"error": str(e)}, status_code=e.status)


@app.get("/api/docs")
def docs_list():
    return documents.list_docs()


@app.post("/api/docs")
async def docs_create(request: Request):
    return _doc_call(documents.create, await request.json())


@app.get("/api/docs/{doc_id}")
def docs_read(doc_id: str):
    return _doc_call(documents.read, doc_id)


@app.put("/api/docs/{doc_id}")
async def docs_update(doc_id: str, request: Request):
    return _doc_call(documents.update, doc_id, await request.json())


@app.delete("/api/docs/{doc_id}")
def docs_delete(doc_id: str):
    r = _doc_call(documents.delete, doc_id)
    return r if r is not None else {"ok": True}


def _signpost_specs(e: Engine) -> list[dict]:
    return [{"id": s.id, "concept": s.concept, "anti": s.anti, "strength": s.strength, "ttl": s.ttl} for s in e.signposts]


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    engine.reload_dials()
    await sock.send_json({"op": "ready", "model": MODEL_ID, "device": engine.device, "axes": engine.axes()})
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    tasks: set[asyncio.Task] = set()

    outq: asyncio.Queue = asyncio.Queue()  # single writer keeps messages ordered

    async def sender():
        while True:
            msg = await outq.get()
            try:
                await sock.send_json(msg)
            except Exception:
                return

    sender_task = asyncio.create_task(sender())

    async def send(msg):
        outq.put_nowait(msg)

    async def handle(msg: dict):
        op, rid = msg.get("op"), msg.get("id")
        async with lock:
            e = engine
            if op in ("axes", "warm"):  # which moods have programmed dials (hot-reloaded)
                e.reload_dials()
                await send({"op": "axes", "axes": e.axes()})
                return
            e.temperature = float(msg.get("temp", e.temperature))
            if "signposts" in msg:
                await asyncio.to_thread(e.set_signposts, msg["signposts"])
            await asyncio.to_thread(e.sync, msg.get("prefix", ""))

            if op == "candidates":
                items = await asyncio.to_thread(e.candidates, int(msg.get("k", 9)))
                await send({"op": "candidates", "id": rid, "items": [c.__dict__ for c in items]})

            elif op == "generate":
                cancel.clear()
                until = msg.get("until")
                n = int(msg.get("n", 8))

                def on_token(piece):
                    loop.call_soon_threadsafe(outq.put_nowait, {"op": "token", "id": rid, "text": piece})

                def stop(piece):
                    if cancel.is_set():
                        return True
                    if until == "sentence":
                        s = piece.rstrip(" ")
                        return s.endswith((".", "!", "?", "\n", ".\"", "?\"", "!\""))
                    return False

                written = await asyncio.to_thread(e.generate, n if not until else 120, on_token, stop)
                await send({"op": "done", "id": rid, "written": written, "signposts": _signpost_specs(e)})

    try:
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("op") == "stop":
                cancel.set()
                continue
            t = asyncio.create_task(handle(msg))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        cancel.set()
        sender_task.cancel()


@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        return JSONResponse({"error": "frontend not built: cd frontend && pnpm install && pnpm build"}, status_code=503)
    return FileResponse(DIST / "index.html")


if DIST.exists():
    # after all API routes, so /api and /ws win
    app.mount("/", StaticFiles(directory=DIST), name="frontend")


def main(argv=None):
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    engine = Engine(args.model, seed=args.seed)
    import uvicorn

    print(f"cosign-web on http://{args.host}:{args.port}  ({engine.device})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
