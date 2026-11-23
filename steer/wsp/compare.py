"""Head-to-head on held-out prompts: no steering vs one or more dial files.

For each prompt, alpha and seed, every method continues the same prompt with
the same seed. The prompt is always fed with no steering active (the prompt's
cache is computed under whatever is active at reset time, which contaminated
the first comparison). Writes compare.json, a label-shuffled
compare_blind.json for a judge, and compare_key.json.

  uv run python -m steer.wsp.compare --run tense-calm-v2 \\
      --dial v1=wsp_runs/tense-calm/earlyW_v1.pt --dial v2=moods/tense__calm.pt --alphas 0.4 0.7 1.0

The word-vector arm of the first comparison went away with word-vector
steering; its results stay in wsp_runs/tense-calm/compare.json.
"""

import argparse
import json
import random
from pathlib import Path

from ..dials import Dial
from ..engine import Engine
from .lab import ROOT, RUNS

HELD_OUT = [
    "The kettle had just started to whistle when",
    "On Sunday mornings my father liked to",
    "The library was almost empty, and the librarian",
    "We drove out to the orchard in late September, and",
    "Mara set the last plate on the table and",
    "The ferry crossing was usually the quietest part of the trip, but",
    "He unlocked the shop at seven, as he always did, and",
    "The garden smelled of cut grass after the rain, and",
    "My sister called to say she was coming home early, so",
    "The old dog lay by the fire while the children",
    "At the end of the pier, a man was",
    "The meeting was scheduled for noon, and Priya",
]


def run(dials: dict[str, Path], alphas, seeds, n_tokens, out_name):
    e = Engine(seed=0)
    e.rack.clear()
    e.dials = {}
    e.reload_dials = lambda: None          # this script drives the rack itself
    loaded = {name: Dial.load(p) for name, p in dials.items()}
    methods = ["base", *loaded]
    rows = []
    for p in HELD_OUT:
        for a in alphas:
            for seed in seeds:
                row = {"prompt": p, "alpha": a, "seed": seed}
                for method in methods:
                    e.rack.clear()
                    e.reset(p)                 # prompt fed with no steering
                    if method != "base":
                        e.rack.add(method, loaded[method])
                        e.rack.set({method: a})
                    e._steer_version += 1
                    e._memo = None
                    e.gen.manual_seed(seed)
                    n0 = len(e.text())
                    e.generate(n_tokens)
                    row[method] = e.text()[n0:]
                e.rack.clear()
                rows.append(row)
                print(f"[{p[:30]}… α={a} seed={seed}]", *(f"\n  {m}: {row[m]!r}" for m in methods), flush=True)
    out = RUNS / out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(rows, indent=2))
    rng = random.Random(0)
    blind, key = [], []
    for i, r in enumerate(rows):
        ms = methods[:]
        rng.shuffle(ms)
        letters = "ABCDEFG"[: len(ms)]
        blind.append({"id": i, "prompt": r["prompt"], **{L: r[m] for L, m in zip(letters, ms)}})
        key.append({"id": i, "alpha": r["alpha"], **{L: m for L, m in zip(letters, ms)}})
    (out / "compare_blind.json").write_text(json.dumps(blind, indent=2))
    (out / "compare_key.json").write_text(json.dumps(key, indent=2))
    print(f"wrote {out}/compare.json, compare_blind.json, compare_key.json ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", action="append", required=True, help="name=path/to/dial.pt (repeatable)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.4, 0.7, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    dials = {}
    for d in a.dial:
        name, path = d.split("=", 1)
        dials[name] = Path(path) if Path(path).is_absolute() else ROOT / path
    run(dials, a.alphas, a.seeds, a.tokens, a.run)


if __name__ == "__main__":
    main()
