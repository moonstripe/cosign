"""Per-token steerable co-writing engine for LFM3-1.7B.

The model is used as a raw text continuer (no chat template, so no
"helpful assistant" persona). The human and the model share one document.
At every token the human can:

  - give the model a token budget and let it write that many tokens
  - inspect the top-k next-token candidates and pick one
  - write text themselves
  - rewind any number of tokens
  - turn mood dials: weight-space edits written by the wsp programmer
    (steer/wsp, moods/*.pt), each a low-rank delta W + alpha*A*B^T applied by
    hooks, so alpha can change from one token to the next

Every token remembers who authored it (model / picked / human), so the
document carries its own provenance.

Cache invariant: `self.cache` always holds the KV/conv state for
[bos] + ids[:-1]. The last token is fed fresh on every peek, so a change of
signpost takes effect on the very next token.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from . import dials as dials_mod

hf_logging.set_verbosity(hf_logging.CRITICAL)
warnings.filterwarnings("ignore")

MODEL_ID = "LiquidAI/LFM3-1.7B"
# Pin the remote modeling code we've actually run (trust_remote_code).
REVISIONS = {"LiquidAI/LFM3-1.7B": "1da06bec6a819da10b54c67e5c90972b46d81871"}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Signpost:
    concept: str
    anti: str | None
    strength: float
    ttl: int | None  # tokens remaining, None = until cleared
    id: str | None = None  # client-side identity, echoed back
    dial: tuple | None = None  # key of the dial in moods/ that implements it
    sign: float = 1.0          # -1 when the dial is run backwards (calm on a tense__calm dial)

    def label(self) -> str:
        s = self.concept + (f" - {self.anti}" if self.anti else "")
        s += f" @{self.strength:g}"
        if self.ttl is not None:
            s += f" ({self.ttl} left)"
        return s


@dataclass
class Candidate:
    token_id: int
    text: str
    prob: float          # probability under the current (steered) distribution
    base_prob: float | None = None  # probability without signposts


class Engine:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        seed: int | None = None,
    ):
        self.device = device or pick_device()
        rev = REVISIONS.get(model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.bos = self.tok.bos_token_id
        self.eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|endoftext|>")} - {None}
        self.layers = self.model.model.layers
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)

        self.signposts: list[Signpost] = []
        # weight-space dials built by the wsp lab; reloaded when moods/ changes.
        # Steering exists ONLY through dials (no word-vector signposts).
        self.rack = dials_mod.DialRack(self.model)
        self.dials: dict = {}
        self._dials_mtime = None
        self.reload_dials()

        self.ids: list[int] = []
        self.src: list[str] = []
        self.cache = None
        self._memo = None
        self._steer_version = 0

    # ------------------------------------------------------------------ steering
    def reload_dials(self):
        d = dials_mod.MOODS_DIR
        mt = max((p.stat().st_mtime for p in d.glob("*.pt")), default=0) if d.exists() else 0
        if mt == self._dials_mtime:
            return
        self._dials_mtime = mt
        self.rack.clear()
        self.dials = dials_mod.load_all()
        for k, dial in self.dials.items():
            self.rack.add(k, dial)
        if self.dials:
            print(f"[engine] dials: {', '.join(f'{c} vs {a}' for c, a in self.dials)}")

    def axes(self) -> list[dict]:
        """Dial-backed mood axes, for the client."""
        return [{"concept": c, "anti": a, "notes": d.notes} for (c, a), d in self.dials.items()]

    def dial_for(self, concept: str, anti: str | None) -> tuple[tuple, float] | None:
        """(dial key, sign) for a mood: (tense, calm) -> forward; (calm, tense) -> reversed."""
        if (concept, anti) in self.dials:
            return (concept, anti), 1.0
        for (c, a) in self.dials:
            if c == anti and a == concept or (anti is None and a == concept):
                return (c, a), -1.0
        return None

    def _sync_dials(self):
        """Dial positions from signpost strengths: strength 10 == alpha 1."""
        alphas: dict = {}
        for s in self.signposts:
            if s.dial:
                alphas[s.dial] = alphas.get(s.dial, 0.0) + s.sign * s.strength / 10
        self.rack.set({k: max(-1.0, min(1.0, a)) for k, a in alphas.items()})

    def set_signposts(self, specs: list[dict]):
        """Replace all signposts. specs: [{concept, anti?, strength (0-10)?, ttl?}].
        Moods without a dial are ignored (returned for the client to report)."""
        self.reload_dials()
        out, ignored = [], []
        for d in specs:
            c, a = d.get("concept", "").strip(), d.get("anti") or None
            if not c:
                continue
            hit = self.dial_for(c, a)
            if not hit:
                ignored.append(c)
                continue
            out.append(Signpost(c, a, float(d.get("strength", 4.0)), d.get("ttl"), d.get("id"), hit[0], hit[1]))
        self.signposts = out
        self._sync_dials()
        self._steer_version += 1
        return ignored

    def add_signpost(self, concept: str, anti: str | None = None, strength: float = 4.0, ttl: int | None = None):
        specs = [{"concept": s.concept, "anti": s.anti, "strength": s.strength, "ttl": s.ttl, "id": s.id} for s in self.signposts]
        self.set_signposts(specs + [{"concept": concept, "anti": anti, "strength": strength, "ttl": ttl}])

    def clear_signposts(self):
        self.signposts.clear()
        self._sync_dials()
        self._steer_version += 1

    def _tick_signposts(self):
        changed = False
        for s in self.signposts:
            if s.ttl is not None:
                s.ttl -= 1
                changed = True
        before = len(self.signposts)
        self.signposts = [s for s in self.signposts if s.ttl is None or s.ttl > 0]
        if changed or len(self.signposts) != before:
            self._sync_dials()
            self._steer_version += 1

    # --------------------------------------------------------------- cache mgmt
    @property
    def _full(self) -> list[int]:
        return [self.bos] + self.ids

    @torch.no_grad()
    def _feed(self, ids: list[int], cache):
        if not ids:
            return None, cache
        out = self.model(
            torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True
        )
        return out.logits[0, -1].float(), out.past_key_values

    def _rebuild(self):
        """Recompute cache for [bos]+ids[:-1] (used after rewind)."""
        self.cache = None
        _, self.cache = self._feed(self._full[:-1], None)
        self._memo = None

    @torch.no_grad()
    def _peek(self):
        """Logits for the next token under current signposts (memoized)."""
        key = (len(self.ids), self._steer_version)
        if self._memo and self._memo[0] == key:
            return self._memo[1], self._memo[2]
        c = copy.deepcopy(self.cache) if self.cache is not None else None
        logits, c = self._feed(self._full[-1:], c)
        self._memo = (key, logits, c)
        return logits, c

    @torch.no_grad()
    def _base_logits(self):
        """Unsteered next-token logits (for showing how a signpost bends things)."""
        c = copy.deepcopy(self.cache) if self.cache is not None else None
        self.rack.enabled = False
        try:
            logits, _ = self._feed(self._full[-1:], c)
        finally:
            self.rack.enabled = True
        return logits

    def _commit(self, token_id: int, source: str):
        _, c = self._peek()
        self.cache = c  # now holds [bos]+ids (old last token included)
        self.ids.append(token_id)
        self.src.append(source)
        self._memo = None
        self._tick_signposts()

    # ------------------------------------------------------------------ actions
    def _shape(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        if self.repetition_penalty != 1.0 and self.ids:
            recent = torch.tensor(sorted(set(self.ids[-64:])), device=logits.device)
            vals = logits[recent]
            logits[recent] = torch.where(vals > 0, vals / self.repetition_penalty, vals * self.repetition_penalty)
        return logits

    def candidates(self, k: int = 8, with_base: bool = True) -> list[Candidate]:
        logits, _ = self._peek()
        probs = F.softmax(self._shape(logits) / max(self.temperature, 1e-5), -1)
        base = None
        if with_base and self.signposts:
            base = F.softmax(self._shape(self._base_logits()) / max(self.temperature, 1e-5), -1)
        p, idx = probs.topk(k)
        return [
            Candidate(int(i), self.tok.decode([int(i)]), float(pp), float(base[i]) if base is not None else None)
            for pp, i in zip(p.tolist(), idx.tolist())
        ]

    def sample(self) -> int:
        logits = self._shape(self._peek()[0])
        if self.temperature <= 0:
            return int(logits.argmax())
        logits = logits / self.temperature
        if self.top_k:
            kth = logits.topk(self.top_k).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
        probs = F.softmax(logits, -1).cpu()
        return int(torch.multinomial(probs, 1, generator=self.gen))

    def generate(self, n: int, on_token=None, stop=None) -> int:
        """Let the model write up to n tokens. `stop(piece)` -> True ends early.
        Returns the number of tokens written."""
        for i in range(n):
            t = self.sample()
            if t in self.eos:
                return i
            self._commit(t, "model")
            piece = self.tok.decode([t])
            if on_token:
                on_token(piece)
            if stop and stop(piece):
                return i + 1
        return n

    def pick(self, token_id: int):
        self._commit(token_id, "picked")

    def write(self, text: str):
        """Human-authored text appended to the shared document."""
        new = self.tok.encode(text, add_special_tokens=False)
        if not new:
            return
        # Advance cache over old last token + all but the last new token.
        _, self.cache = self._feed(self._full[-1:] + new[:-1], self.cache)
        self.ids.extend(new)
        self.src.extend(["human"] * len(new))
        self._memo = None

    def rewind(self, n: int):
        n = min(n, len(self.ids))
        if n <= 0:
            return
        del self.ids[-n:]
        del self.src[-n:]
        self._rebuild()

    def sync(self, text: str):
        """Make the document equal `text`, keeping as much cache as possible.
        Used by editors where the human can change anything anywhere."""
        acc, k = "", 0
        for t in self.ids:
            p = self.tok.decode([t])
            if not text.startswith(p, len(acc)):
                break
            acc += p
            k += 1
        # Step back if the kept prefix doesn't decode cleanly, or if new text
        # glues onto the last kept token (so "cat"+"s" re-tokenizes as "cats").
        while k and (self.tok.decode(self.ids[:k]) != text[: len(self.tok.decode(self.ids[:k]))]):
            k -= 1
        rest = text[len(self.tok.decode(self.ids[:k])):]
        if k and rest and not rest[0].isspace():
            k -= 1
        kept = self.tok.decode(self.ids[:k]) if k else ""
        if k < len(self.ids):
            self.rewind(len(self.ids) - k)
        if text[len(kept):]:
            self.write(text[len(kept):])

    def reset(self, text: str = ""):
        self.ids, self.src = [], []
        self.cache, self._memo = None, None
        if text:
            self.write(text)

    # ----------------------------------------------------------------- reading
    def text(self) -> str:
        return self.tok.decode(self.ids)

    def spans(self) -> list[tuple[str, str]]:
        """Contiguous (source, text) runs, for provenance display."""
        out: list[tuple[str, list[int]]] = []
        for t, s in zip(self.ids, self.src):
            if out and out[-1][0] == s:
                out[-1][1].append(t)
            else:
                out.append((s, [t]))
        return [(s, self.tok.decode(ts)) for s, ts in out]

    def provenance(self) -> dict[str, float]:
        if not self.ids:
            return {}
        counts: dict[str, int] = {}
        for s in self.src:
            counts[s] = counts.get(s, 0) + 1
        return {k: v / len(self.src) for k, v in counts.items()}
