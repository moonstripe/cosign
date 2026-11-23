"""Dials: weight-space edits that can be turned up and down per token.

A dial is what the weight-space programmer delivers for one mood axis
(e.g. tense vs calm). It is a list of low-rank weight deltas

    W' = W + alpha * A @ B.T          (A: [out, r], B: [in, r])

on named nn.Linear modules, where alpha in [0, 1] is the dial position
(1 = the strongest setting the programmer judged still fluent). Instead of
rewriting the weights, the delta is applied with a forward hook,

    y += alpha * (x @ B) @ A.T

so alpha can change from one token to the next (the wheel), and alpha = 0
is exactly the original model. `bake()` writes it into the weights instead,
for exporting a permanently edited checkpoint.

File format (torch.save): moods/<concept>__<anti>.pt
    {"concept": str, "anti": str, "deltas": [{"module": str, "A": Tensor, "B": Tensor}],
     "notes": str, "created": str}
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

MOODS_DIR = Path(os.environ.get("WSP_MOODS", Path(__file__).resolve().parent.parent / "moods"))


def dial_path(concept: str, anti: str | None, root: Path = MOODS_DIR) -> Path:
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")  # noqa: E731
    return root / f"{slug(concept)}__{slug(anti or 'none')}.pt"


@dataclass
class Delta:
    module: str
    A: torch.Tensor  # [out, r]
    B: torch.Tensor  # [in, r]


@dataclass
class Dial:
    concept: str
    anti: str | None
    deltas: list[Delta]
    notes: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_weights_touched(self) -> int:
        """Entries of the weight matrices that a nonzero alpha changes."""
        n = 0
        for d in self.deltas:
            rows = int((d.A.abs().sum(1) > 0).sum())
            cols = int((d.B.abs().sum(1) > 0).sum())
            n += rows * cols
        return n

    def save(self, path: Path | None = None) -> Path:
        path = path or dial_path(self.concept, self.anti)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "concept": self.concept, "anti": self.anti, "notes": self.notes, "meta": self.meta,
            "deltas": [{"module": d.module, "A": d.A.float().cpu(), "B": d.B.float().cpu()} for d in self.deltas],
        }, path)
        return path

    @staticmethod
    def load(path: Path) -> "Dial":
        o = torch.load(path, map_location="cpu", weights_only=False)
        return Dial(o["concept"], o.get("anti"), [Delta(d["module"], d["A"], d["B"]) for d in o["deltas"]],
                    o.get("notes", ""), o.get("meta", {}))


def load_all(root: Path = MOODS_DIR) -> dict[tuple[str, str | None], Dial]:
    out = {}
    for p in sorted(root.glob("*.pt")) if root.exists() else []:
        try:
            d = Dial.load(p)
            out[(d.concept, d.anti)] = d
        except Exception as e:  # a broken file shouldn't take the server down
            print(f"[dials] skipping {p.name}: {e}")
    return out


class DialRack:
    """Hooks that apply any number of dials at their current alphas.

    `set({key: alpha})` changes positions; the hooks read them on every
    forward pass. `enabled = False` gives the original model (for the
    unsteered comparison) without removing hooks.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.modules = dict(model.named_modules())
        self.dials: dict[tuple, Dial] = {}
        self.alpha: dict[tuple, float] = {}
        self.enabled = True
        self._hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._by_module: dict[str, list[tuple[tuple, torch.Tensor, torch.Tensor]]] = {}

    def add(self, key: tuple, dial: Dial):
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.remove(key)
        self.dials[key] = dial
        for d in dial.deltas:
            if d.module not in self.modules:
                raise KeyError(f"dial {key}: no module named {d.module}")
            self._by_module.setdefault(d.module, []).append((key, d.A.to(dev, dtype), d.B.to(dev, dtype)))
            if d.module not in self._hooks:
                self._hooks[d.module] = self.modules[d.module].register_forward_hook(self._hook(d.module))

    def remove(self, key: tuple):
        if key not in self.dials:
            return
        del self.dials[key]
        self.alpha.pop(key, None)
        for m in list(self._by_module):
            self._by_module[m] = [e for e in self._by_module[m] if e[0] != key]
            if not self._by_module[m]:
                del self._by_module[m]
                self._hooks.pop(m).remove()

    def set(self, alphas: dict):
        """alpha per dial: a float, or a tensor [batch] for a different alpha per row."""
        out = {}
        for k, v in alphas.items():
            if k not in self.dials:
                continue
            if torch.is_tensor(v):
                if bool((v != 0).any()):
                    out[k] = v
            elif v:
                out[k] = float(v)
        self.alpha = out

    def _hook(self, name: str):
        def hook(module, args, out):
            if not self.enabled or not self.alpha:
                return out
            x = args[0]
            for key, A, B in self._by_module.get(name, []):
                a = self.alpha.get(key)
                if a is None:
                    continue
                if torch.is_tensor(a):                       # per-row alphas, x: [B, T, in]
                    a = a.to(out.device, out.dtype).view(-1, *([1] * (out.dim() - 1)))
                out = out + a * ((x @ B) @ A.T)
            return out
        return hook

    def clear(self):
        for h in self._hooks.values():
            h.remove()
        self._hooks.clear()
        self._by_module.clear()
        self.dials.clear()
        self.alpha.clear()


@torch.no_grad()
def bake(model: nn.Module, dial: Dial, alpha: float):
    """Write alpha * delta into the weights (irreversible; for export)."""
    mods = dict(model.named_modules())
    for d in dial.deltas:
        W: torch.Tensor = mods[d.module].weight  # type: ignore[assignment]
        W += alpha * d.A.to(W.device, W.dtype) @ d.B.to(W.device, W.dtype).T
