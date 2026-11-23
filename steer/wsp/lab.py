"""Weight-space programming lab for LFM3-1.7B.

Modelled on the harness in "Sparks of Weight-Space Programming" (Takehi,
Teoh, Zhang, Khattab; MIT CSAIL 2026), Experiment 8. A programmer (an LLM
agent) gets a natural-language spec — "build a dial that makes the model
write tensely rather than calmly" — and two tools:

  1. a persistent Python kernel over the weights (read tensors, compute,
     save candidate dials). The model object is NOT in the kernel: it can
     read weights but not run the network.
  2. a forward-pass tool with a fixed budget. One pass is either a batch of
     generations (original and/or candidate dials, paired by seed) or a
     teacher-forced read of up to 256 texts (log-probs + activations).

No gradients, no training, no optimisers. Every call is logged to
wsp_runs/<run>/ (tool_calls.jsonl, requests/, work/), like the paper's logs.

    uv run wsp-lab --run tense-calm --budget 15      # starts the lab on :8790
    uv run wsp status | kernel | forward              # the programmer's CLI
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..dials import MOODS_DIR, Delta, Dial, DialRack, dial_path
from ..engine import MODEL_ID, REVISIONS, pick_device

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = Path(os.environ.get("WSP_RUNS", ROOT / "wsp_runs"))  # overridable (Modal volume)

LIMITS = {
    "generate": {"prompts": 16, "samples": 16, "settings": 4, "max_tokens": 96, "sequences": 256},
    "read": {"items": 256, "settings": 3, "mlp_layers": 6, "tokens_per_item": 256},
    "kernel_output_chars": 20000,
}

RULES = """\
RULES (enforced by the lab or audited from the logs afterwards)
- No gradient descent or training of any kind: no backward passes, autograd, optimisers,
  or iterative fitting of weights to a loss.
- You cannot run the model in the kernel. It is not there. The forward tool is the only
  way to run it, and every call costs one pass from the budget (failed requests are free).
- Do not touch the embedding, the output head or the final norm (save_dial refuses them).
- Deliver dials (low-rank deltas on nn.Linear weights): W' = W + alpha * A @ B.T, alpha in [0, 1].
  alpha = 1 must be the strongest setting you judge still fluent; alpha = 0 is the original model.
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


class Lab:
    def __init__(self, run: str, budget: int, device: str | None = None):
        self.device = device or pick_device()
        rev = REVISIONS.get(MODEL_ID)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=rev, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device).eval()
        self.cfg = self.model.config
        self.n_layers = self.cfg.num_hidden_layers
        self.layers = self.model.model.layers
        self.bos = self.tok.bos_token_id
        self.rack = DialRack(self.model)
        self.candidates: dict[str, Dial] = {}
        self._lock = threading.Lock()   # one forward pass / kernel call at a time (pass numbering, shared rack)

        self.run = run
        self.dir = RUNS / run
        (self.dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.dir / "work").mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.budget = st.get("budget", budget)
        self.used = st.get("used", 0)
        self.kernel_runs = st.get("kernel_runs", 0)
        self.submitted = st.get("submitted", [])
        for p in sorted((self.dir / "work").glob("cand_*.pt")):   # resume candidates
            self.candidates[p.stem[5:]] = Dial.load(p)
        self.ns = self._namespace()
        self._save_state()

    # ------------------------------------------------------------------ bookkeeping
    def _save_state(self):
        self.state_path.write_text(json.dumps({
            "run": self.run, "model": MODEL_ID, "budget": self.budget, "used": self.used,
            "kernel_runs": self.kernel_runs, "candidates": sorted(self.candidates), "submitted": self.submitted,
            "updated": _now(),
        }, indent=2))

    def _log(self, rec: dict):
        with open(self.dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"t": _now(), **rec}) + "\n")

    def status(self) -> dict:
        return {
            "run": self.run, "budget": self.budget, "passes_used": self.used, "passes_left": self.budget - self.used,
            "kernel_runs": self.kernel_runs, "candidates": {k: self._dial_summary(d) for k, d in self.candidates.items()},
            "submitted": self.submitted, "limits": LIMITS, "rules": RULES, "architecture": self.architecture(),
        }

    def architecture(self) -> dict:
        c = self.cfg
        return {
            "model": MODEL_ID, "hidden_size": c.hidden_size, "intermediate_size": c.intermediate_size,
            "num_layers": c.num_hidden_layers, "layer_types": list(c.layer_types),
            "block": "x += post_operator_norm(operator(operator_norm(x))); "
                     "x += post_ffn_norm(feed_forward(ffn_norm(x)))  [sandwich RMSNorms]",
            "feed_forward": "w2(silu(w1(x)) * w3(x)); w1,w3: hidden->intermediate, w2: intermediate->hidden",
            "operators": "conv layers: model.layers.i.conv (short convolution); attention layers: model.layers.i.self_attn",
            "decoding": "raw text, no chat template, a <|startoftext|> token is prepended",
        }

    # ------------------------------------------------------------------ kernel
    def _namespace(self) -> dict:
        params = dict(self.model.named_parameters())

        def names(pattern: str = "") -> list[tuple[str, tuple]]:
            """Parameter names and shapes (substring filter)."""
            return [(n, tuple(p.shape)) for n, p in params.items() if pattern in n]

        def load(name: str) -> torch.Tensor:
            """A float32 CPU copy of a weight tensor (editing it changes nothing)."""
            return params[name].detach().float().cpu().clone()

        def acts(k: int) -> dict:
            """Results of forward read number k (log-probs and captured activations)."""
            return torch.load(self.dir / "work" / f"read_{k}.pt", weights_only=False)

        def gens(k: int) -> dict:
            """Results of forward generate number k."""
            return json.loads((self.dir / "work" / f"gen_{k}.json").read_text())

        def save_dial(name: str, concept: str, anti: str, deltas: list, notes: str = "") -> dict:
            """Register a candidate dial. deltas: [(module_name, A [out,r], B [in,r]), ...]
            module_name is an nn.Linear (e.g. 'model.layers.15.feed_forward.w2')."""
            mods = dict(self.model.named_modules())
            ds = []
            for module, A, B in deltas:
                if module not in mods or not isinstance(mods[module], torch.nn.Linear):
                    raise ValueError(f"{module} is not an nn.Linear in the model")
                if any(s in module for s in ("embed", "lm_head", "embedding_norm")):
                    raise ValueError(f"{module}: embeddings, the output head and the final norm are off limits")
                W = mods[module].weight
                A, B = torch.as_tensor(A).float(), torch.as_tensor(B).float()
                if A.dim() == 1: A = A[:, None]
                if B.dim() == 1: B = B[:, None]
                if A.shape[0] != W.shape[0] or B.shape[0] != W.shape[1] or A.shape[1] != B.shape[1]:
                    raise ValueError(f"{module}: W is {tuple(W.shape)}, need A [{W.shape[0]}, r] and B [{W.shape[1]}, r]")
                ds.append(Delta(module, A, B))
            d = Dial(concept, anti, ds, notes, {"run": self.run, "name": name, "saved": _now()})
            d.save(self.dir / "work" / f"cand_{name}.pt")
            self.candidates[name] = d
            self._save_state()
            return self._dial_summary(d)

        def submit_dial(name: str) -> str:
            """Deliver a candidate: it becomes the editor's dial for its mood axis."""
            d = self.candidates[name]
            p = d.save(dial_path(d.concept, d.anti))
            try:
                rel = str(p.relative_to(ROOT))
            except ValueError:          # on Modal the moods dir is a volume outside the repo
                rel = str(p)
            self.submitted.append({"name": name, "path": rel, "t": _now()})
            self._save_state()
            return str(p)

        return {
            "torch": torch, "np": np, "math": math, "json": json, "F": F,
            "names": names, "load": load, "acts": acts, "gens": gens,
            "save_dial": save_dial, "submit_dial": submit_dial, "candidates": self.candidates,
            "WORK": self.dir / "work", "ARCH": self.architecture(), "tokenizer": self.tok,
        }

    def _dial_summary(self, d: Dial) -> dict:
        return {"concept": d.concept, "anti": d.anti, "modules": [x.module for x in d.deltas],
                "ranks": [x.A.shape[1] for x in d.deltas], "weights_touched": d.n_weights_touched}

    def kernel(self, code: str) -> dict:
        with self._lock:
            return self._kernel(code)

    def _kernel(self, code: str) -> dict:
        self.kernel_runs += 1
        buf = io.StringIO()
        t0 = time.time()
        ok = True
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, f"<kernel {self.kernel_runs}>", "exec"), self.ns)
            except Exception:
                ok = False
                traceback.print_exc()
        out = buf.getvalue()
        if len(out) > LIMITS["kernel_output_chars"]:
            out = out[: LIMITS["kernel_output_chars"] // 2] + "\n... [output truncated] ...\n" + out[-LIMITS["kernel_output_chars"] // 2:]
        self._log({"tool": "kernel", "n": self.kernel_runs, "ok": ok, "code": code, "output": out, "secs": round(time.time() - t0, 2)})
        self._save_state()
        return {"ok": ok, "output": out}

    # ------------------------------------------------------------------ forward passes
    def _settings(self, settings: list[dict] | None, max_n: int) -> list[dict]:
        settings = settings or [{"dial": None, "alpha": 0}]
        if len(settings) > max_n:
            raise ValueError(f"at most {max_n} settings per pass")
        for s in settings:
            if s.get("dial") and s["dial"] not in self.candidates:
                raise ValueError(f"unknown candidate dial {s['dial']!r}; have {sorted(self.candidates)}")
        return settings

    @contextlib.contextmanager
    def _apply(self, setting: dict):
        name, alpha = setting.get("dial"), float(setting.get("alpha", 1.0))
        self.rack.clear()
        if name and alpha:
            self.rack.add(("cand", name), self.candidates[name])
            self.rack.set({("cand", name): alpha})
        try:
            yield
        finally:
            self.rack.clear()

    @torch.no_grad()
    def _sample(self, prompt: str, n: int, steps: int, temp: float, top_k: int, seed: int) -> list[str]:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.tensor([[self.bos] + self.tok.encode(prompt, add_special_tokens=False)] * n, device=self.device)
        out = self.model(input_ids=x, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1].float()
        toks = []
        for _ in range(steps):
            l = logits / max(temp, 1e-5)
            if top_k:
                kth = l.topk(top_k, -1).values[:, -1:]
                l = l.masked_fill(l < kth, -float("inf"))
            nxt = torch.multinomial(F.softmax(l, -1).cpu(), 1, generator=g).to(self.device)
            toks.append(nxt)
            out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1].float()
        t = torch.cat(toks, 1).tolist()
        eos = self.tok.eos_token_id
        return [self.tok.decode(r[: r.index(eos)] if eos in r else r) for r in t]

    @torch.no_grad()
    def _sample_settings(self, prompt, settings, n, steps, temp, top_k, seed) -> list[list[str]]:
        S = len(settings)
        rows = S * n
        names = sorted({s["dial"] for s in settings if s.get("dial") and float(s.get("alpha", 1.0))})
        self.rack.clear()
        for name in names:
            self.rack.add(("cand", name), self.candidates[name])
        self.rack.set({("cand", name): torch.tensor(
            [float(s.get("alpha", 1.0)) if s.get("dial") == name else 0.0 for s in settings for _ in range(n)],
            device=self.device) for name in names})
        try:
            g = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.tensor([[self.bos] + self.tok.encode(prompt, add_special_tokens=False)] * rows, device=self.device)
            out = self.model(input_ids=x, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1].float()
            toks = []
            for _ in range(steps):
                l = logits / max(temp, 1e-5)
                if top_k:
                    kth = l.topk(top_k, -1).values[:, -1:]
                    l = l.masked_fill(l < kth, -float("inf"))
                cum = F.softmax(l, -1).cumsum(-1)
                u = torch.rand(n, generator=g).repeat(S).to(self.device)        # same draw for paired rows
                nxt = torch.searchsorted(cum, (u * cum[:, -1])[:, None]).clamp_(max=cum.shape[-1] - 1)
                toks.append(nxt)
                out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
                cache, logits = out.past_key_values, out.logits[:, -1].float()
        finally:
            self.rack.clear()
        t = torch.cat(toks, 1).tolist()
        eos = self.tok.eos_token_id
        texts = [self.tok.decode(r[: r.index(eos)] if eos in r else r) for r in t]
        return [texts[si * n:(si + 1) * n] for si in range(S)]

    def _generate(self, req: dict) -> dict:
        L = LIMITS["generate"]
        prompts = req["prompts"]
        samples = int(req.get("samples", 4))
        steps = min(int(req.get("max_tokens", 60)), L["max_tokens"])
        settings = self._settings(req.get("settings"), L["settings"])
        if len(prompts) > L["prompts"] or samples > L["samples"] or len(prompts) * samples * len(settings) > L["sequences"]:
            raise ValueError(f"generate limits: {L}")
        temp, top_k, seed = float(req.get("temperature", 0.8)), int(req.get("top_k", 50)), int(req.get("seed", 0))
        # all settings of one prompt run as ONE batch (rows = settings x samples), with
        # per-row dial alphas and shared random draws, so sample i is paired across settings
        per_prompt = [self._sample_settings(p, settings, samples, steps, temp, top_k, seed + 1000 * pi)
                      for pi, p in enumerate(prompts)]
        results = [{"setting": s, "outputs": [{"prompt": p, "samples": per_prompt[pi][si]} for pi, p in enumerate(prompts)]}
                   for si, s in enumerate(settings)]
        return {"kind": "generate", "note": "same seeds across settings: sample i of each setting is paired", "results": results}

    @torch.no_grad()
    def _read(self, req: dict) -> dict:
        L = LIMITS["read"]
        items = req["items"]
        settings = self._settings(req.get("settings"), L["settings"])
        mlp_layers = list(req.get("mlp_layers", []))
        if len(items) > L["items"] or len(mlp_layers) > L["mlp_layers"]:
            raise ValueError(f"read limits: {L}")
        cap: dict = {}
        hooks = []
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(("res", i), o[0].float())))
        for i in mlp_layers:
            hooks.append(self.layers[i].feed_forward.w2.register_forward_pre_hook(
                lambda m, a, i=i: cap.__setitem__(("mlp", i), a[0][0].float())))
        summary, saved = [], {"items": items, "settings": settings, "mlp_layers": mlp_layers}
        try:
            for si, s in enumerate(settings):
                lp_mean, lp_tok, res_mean, mlp_mean = [], [], [], {i: [] for i in mlp_layers}
                with self._apply(s):
                    for it in items:
                        pre = [self.bos] + self.tok.encode(it["prefix"], add_special_tokens=False)
                        cont = self.tok.encode(it["continuation"], add_special_tokens=False)[: L["tokens_per_item"]]
                        if not cont:
                            raise ValueError("empty continuation")
                        ids = torch.tensor([pre + cont], device=self.device)
                        cap.clear()
                        logits = self.model(input_ids=ids, use_cache=False).logits[0].float()
                        sl = slice(len(pre) - 1, len(pre) + len(cont) - 1)       # positions predicting cont
                        lp = F.log_softmax(logits[sl], -1).gather(1, torch.tensor(cont, device=self.device)[:, None])[:, 0]
                        lp_tok.append(lp.cpu())
                        lp_mean.append(lp.mean().item())
                        cs = slice(len(pre), len(pre) + len(cont))               # positions OF cont tokens
                        res_mean.append(torch.stack([cap[("res", i)][cs].mean(0) for i in range(self.n_layers)]).cpu())
                        for i in mlp_layers:
                            mlp_mean[i].append(cap[("mlp", i)][cs].mean(0).cpu())
                saved[si] = {
                    "logp_mean": torch.tensor(lp_mean), "logp_tokens": lp_tok,
                    "residual_mean": torch.stack(res_mean).half(),                # [items, layers, hidden]
                    "mlp_in_mean": {i: torch.stack(v).half() for i, v in mlp_mean.items()},  # w2 inputs [items, 7680]
                }
                summary.append({"setting": s, "logp_mean_per_item": [round(x, 4) for x in lp_mean],
                                "logp_mean_overall": round(float(np.mean(lp_mean)), 4)})
        finally:
            for h in hooks:
                h.remove()
        return {"kind": "read", "summary": summary, "_saved": saved,
                "note": "activations: acts(k)[setting_index] -> residual_mean [items, 24 layers, 2048] (decoder layer "
                        "outputs, mean over continuation tokens), mlp_in_mean[layer] [items, 7680] (inputs to w2), "
                        "logp_mean, logp_tokens"}

    def forward(self, req: dict) -> dict:
        with self._lock:
            return self._forward(req)

    def _forward(self, req: dict) -> dict:
        kind = req.get("kind")
        if self.used >= self.budget:
            return {"ok": False, "error": f"budget exhausted ({self.used}/{self.budget} passes used)"}
        t0 = time.time()
        k = self.used + 1
        try:
            if kind == "generate":
                res = self._generate(req)
            elif kind == "read":
                res = self._read(req)
            else:
                raise ValueError("kind must be 'generate' or 'read'")
        except Exception as e:      # failed requests do not count
            self._log({"tool": "forward", "ok": False, "request": req, "error": repr(e)})
            return {"ok": False, "error": repr(e), "note": "failed requests do not count against the budget"}
        self.used = k
        (self.dir / "requests" / f"fwd_{k}.json").write_text(json.dumps(req, indent=2))
        if kind == "read":
            torch.save(res.pop("_saved"), self.dir / "work" / f"read_{k}.pt")
        else:
            (self.dir / "work" / f"gen_{k}.json").write_text(json.dumps(res, indent=2))
        res.update({"ok": True, "pass": k, "passes_left": self.budget - self.used, "secs": round(time.time() - t0, 1),
                    "load_in_kernel": f"{'acts' if kind == 'read' else 'gens'}({k})"})
        (self.dir / "requests" / f"fwd_{k}.done.json").write_text(json.dumps(res, indent=2))
        self._log({"tool": "forward", "ok": True, "pass": k, "kind": kind, "request": req, "secs": res["secs"]})
        self._save_state()
        return res


# ---------------------------------------------------------------------- server
app = FastAPI()
lab: Lab | None = None


class KernelReq(BaseModel):
    code: str


@app.get("/status")
def status():
    return lab.status()


@app.post("/kernel")
def kernel(req: KernelReq):
    return lab.kernel(req.code)


@app.post("/forward")
def forward(req: dict):
    r = lab.forward(req)
    return JSONResponse(r, status_code=200 if r.get("ok") else 400)


def main(argv=None):
    global lab
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. tense-calm")
    ap.add_argument("--budget", type=int, default=15, help="forward passes")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)
    lab = Lab(args.run, args.budget)
    import uvicorn

    print(f"wsp lab '{args.run}' on http://127.0.0.1:{args.port}  ({lab.device}, {lab.used}/{lab.budget} passes used)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
