"""Run a wsp lab on a Modal GPU, one container per mood axis.

    uv run modal run steer/wsp/modal_lab.py --run playful-grave --budget 15

The container loads LFM3-1.7B on CUDA, starts the same FastAPI lab as
`wsp-lab`, and opens a Modal tunnel. The URL is printed and written to
wsp_runs/<run>/LAB_URL locally. Point the programmer at it with
WSP_LAB=<url> uv run wsp status. Lab logs and submitted dials go to the Modal
Volume "wsp" (/data/wsp_runs, /data/moods). Pull them back with

    uv run python -m steer.wsp.modal_lab pull <run>

The lab stays up until the local `modal run` process is stopped, or until the
function timeout.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
GPU = os.environ.get("WSP_GPU", "L40S")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "transformers>=5.12", "accelerate", "safetensors", "huggingface_hub",
                 "fastapi", "uvicorn[standard]", "httpx", "python-dotenv", "numpy")
    .env({"HF_HOME": "/cache/hf", "WSP_RUNS": "/data/wsp_runs", "WSP_MOODS": "/data/moods",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("steer")
)
hf_cache = modal.Volume.from_name("wsp-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("wsp", create_if_missing=True)
app = modal.App("wsp-lab", image=image)


@app.function(gpu=GPU, volumes={"/cache": hf_cache, "/data": data}, timeout=4 * 3600, max_containers=10)
def lab(run: str, budget: int, url_q: modal.Queue | None = None):
    import threading
    import time

    import uvicorn

    from steer.wsp import lab as L

    t0 = time.time()
    L.lab = L.Lab(run, budget, device="cuda")
    hf_cache.commit()
    print(f"[{run}] model loaded in {time.time() - t0:.0f}s on {GPU}")
    server = uvicorn.Server(uvicorn.Config(L.app, host="0.0.0.0", port=8790, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    with modal.forward(8790) as tunnel:
        print(f"LAB_URL {tunnel.url}", flush=True)
        if url_q is not None:
            url_q.put(tunnel.url)
        while True:                       # keep the container (and the logs volume) alive
            time.sleep(30)
            data.commit()


@app.local_entrypoint()
def main(run: str, budget: int = 15):
    with modal.Queue.ephemeral() as q:
        call = lab.spawn(run, budget, q)
        url = q.get(timeout=900)
        out = ROOT / "wsp_runs" / run
        out.mkdir(parents=True, exist_ok=True)
        (out / "LAB_URL").write_text(url)
        print(f"LAB_URL {url}\n  WSP_LAB={url} uv run wsp status", flush=True)
        try:
            call.get()                    # block; Ctrl-C / kill stops the lab
        except KeyboardInterrupt:
            call.cancel()


def pull(run: str):
    """Copy the lab logs (and any submitted dial) from the volume into the repo."""
    dst = ROOT / "wsp_runs"
    subprocess.run(["modal", "volume", "get", "--force", "wsp", f"wsp_runs/{run}", str(dst)], check=True)
    r = subprocess.run(["modal", "volume", "ls", "wsp", "moods"], capture_output=True, text=True)
    for name in r.stdout.split():
        name = Path(name).name
        if name.endswith(".pt"):
            subprocess.run(["modal", "volume", "get", "--force", "wsp", f"moods/{name}", str(ROOT / "moods")], check=True)
    print("pulled", run)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "pull":
    pull(sys.argv[2])
