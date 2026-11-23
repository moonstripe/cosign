"""Detector experiment: does per-sentence mood steering change Pangram's verdict on LFM3 text?

See PREREG.md (written and committed before any data). Three resumable phases:

    uv run python experiments/detector/run.py gen      # generate all arms -> samples.jsonl
    uv run python experiments/detector/run.py score    # Pangram on each cut, alone -> scores.jsonl
    uv run python experiments/detector/run.py report   # paired stats -> results.md
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
import os
SMOKE = bool(os.environ.get("SMOKE"))   # SMOKE=1: one opening, short texts, separate files, no claims
TAG = "smoke_" if SMOKE else ""
SAMPLES = HERE / f"{TAG}samples.jsonl"
SCORES = HERE / f"{TAG}scores.jsonl"
RESULTS = HERE / f"{TAG}results.md"

LENGTHS = (30,) if SMOKE else (150, 300, 500)
TARGET_WORDS = 45 if SMOKE else 540   # generate a little past the longest cut
MAX_TOKENS = 1100
SEEDS = (0, 1)  # smoke uses both too (cheap)
POS = [("tense", "calm"), ("tender", "cold"), ("playful", "grave"), ("lush", "plain")]
ALL8 = POS + [(b, a) for a, b in POS]   # anti side = same dial, negative alpha
ARMS = ("unsteered", "constant", "vary_pos", "vary_both", "assistant")

NARRATIVE = [
    "The kettle had just started to whistle when",
    "On Sunday mornings my father liked to",
    "The library was almost empty, and the librarian",
    "We drove out to the orchard in late September, and",
    "Mara set the last plate on the table and",
]
ACADEMIC_CHUNKS = (1, 2, 12, 23, 58)
ASSISTANT_PROMPT = ("Continue the following text. Write at least 500 more words in the same style. "
                    "Output only the continuation.\n\n")

QUIZ = re.compile(r"(how many|how much|how long|solve|step[- ]by[- ]step|express your answer|\bcalculate\b|\bA\)|\bB\)|let'?s denote|\\\()", re.I)
WORD = re.compile(r"\S+")


def openings() -> list[dict]:
    save = json.load(open(ROOT / "samples" / "ai_scientist_branch_48pct.cosign.json"))
    chunks = {n["meta"]["chunk"]["i"]: n["inserted"].strip() for n in save["tree"]["nodes"].values() if n["via"] == "import"}
    out = [{"id": f"acad{i}", "register": "academic", "text": chunks[i]} for i in ACADEMIC_CHUNKS]
    out += [{"id": f"narr{k}", "register": "narrative", "text": t} for k, t in enumerate(NARRATIVE)]
    return [out[0], out[5]] if SMOKE else out


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])[\"”’')\]]*\s+|\n+", text)
    return [p for p in parts if p.strip()]


def cut(text: str, target: int) -> str | None:
    """Shortest prefix ending at a sentence end with >= target words (None if the text is too short)."""
    for m in re.finditer(r"[.!?][\"”’')\]]*(?=\s|$)|\n", text):
        prefix = text[: m.end()].strip()
        if len(WORD.findall(prefix)) >= target:
            return prefix
    return None


def repetitive(text: str, n: int = 6, times: int = 3) -> bool:
    w = WORD.findall(text.lower())
    grams = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n]); grams[g] = grams.get(g, 0) + 1
    return any(c >= times for c in grams.values())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ------------------------------------------------------------------ generation
def gen():
    from steer.engine import Engine
    done = {(r["arm"], r["sample"]) for r in load_jsonl(SAMPLES)}
    e = Engine(seed=0)
    assert all(e.dial_for(c, a) for c, a in ALL8), "missing dials"
    tok = e.tok
    ops = openings()
    out = open(SAMPLES, "a")
    for oi, op in enumerate(ops):
        for seed in SEEDS:
            sid = f"{op['id']}-s{seed}"
            seed_val = 1000 * oi + seed
            rng = random.Random(seed_val)
            strengths = [round(rng.uniform(0.25, 0.60), 3) for _ in range(400)]
            const_mood = POS[(oi + seed) % 4]
            for arm in ARMS:
                if (arm, sid) in done:
                    continue
                t0 = time.time()
                e.gen.manual_seed(seed_val)
                mood_rng = random.Random(seed_val * 7 + 1)
                e.set_signposts([])
                if arm == "assistant":
                    prompt = tok.apply_chat_template([{"role": "user", "content": ASSISTANT_PROMPT + op["text"]}],
                                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    e.reset(prompt)
                else:
                    e.reset(op["text"])
                k0 = len(e.ids)
                plan, last, eos = [], None, False
                while True:
                    gen_text = tok.decode(e.ids[k0:])
                    if len(WORD.findall(gen_text)) >= TARGET_WORDS or len(e.ids) - k0 >= MAX_TOKENS:
                        break
                    i = len(plan)
                    mood, s = None, 0.0
                    if arm == "constant":
                        mood, s = const_mood, strengths[i]
                    elif arm in ("vary_pos", "vary_both"):
                        pool = POS if arm == "vary_pos" else ALL8
                        mood = mood_rng.choice([m for m in pool if m != last] or pool)
                        s = strengths[i]
                    if mood:
                        e.set_signposts([{"concept": mood[0], "anti": mood[1], "strength": s * 10}])
                    else:
                        e.set_signposts([])
                    tail, stopped = [""], [False]

                    def stop(piece, tail=tail, stopped=stopped):
                        tail[0] = (tail[0] + piece)[-8:]
                        stopped[0] = "\n" in piece or bool(re.search(r"[.!?][\"”’')\]]*\s*$", tail[0]))
                        return stopped[0]

                    n = e.generate(120, stop=stop)
                    plan.append({"mood": mood and list(mood), "strength": s, "tokens": n})
                    last = mood
                    if not stopped[0] and n < 120:   # the model emitted end-of-text
                        eos = True
                        break
                e.set_signposts([])
                text = tok.decode(e.ids[k0:]).strip()
                rec = {"arm": arm, "sample": sid, "opening": op["id"], "register": op["register"], "seed": seed,
                       "text": text, "words": len(WORD.findall(text)), "sentences_planned": len(plan), "eos": eos,
                       "plan": plan, "quiz": bool(QUIZ.search(text)), "repetitive": repetitive(text),
                       "cuts": {str(L): cut(text, L) for L in LENGTHS}, "secs": round(time.time() - t0, 1)}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"{arm:10s} {sid:9s} {rec['words']:4d} words {rec['secs']:6.1f}s quiz={rec['quiz']} rep={rec['repetitive']}", flush=True)
    human(out)


def _novel_passage(path: Path, frac: float, words: int = TARGET_WORDS) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    start, end = raw.find("*** START OF"), raw.find("*** END OF")
    body = raw[raw.find("\n", start) + 1: end if end > 0 else None]
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if len(p.split()) > 25 and not p.isupper()]
    i = int(len(paras) * frac)
    out = []
    while len(" ".join(out).split()) < words and i < len(paras):
        out.append(paras[i]); i += 1
    return "\n\n".join(out)


def human(out):
    done = {(r["arm"], r["sample"]) for r in load_jsonl(SAMPLES)}
    att = json.load(open(ROOT / "samples" / "attention_2017.cosign.json"))
    ch = sorted([n for n in att["tree"]["nodes"].values() if n["via"] == "import"], key=lambda n: n["meta"]["chunk"]["i"])
    texts = []
    for start in (3, 15, 27, 39, 51):
        acc, i = [], start
        while len(" ".join(acc).split()) < TARGET_WORDS and i < len(ch):
            acc.append(ch[i]["inserted"].strip()); i += 1
        texts.append(("attention", f"att{start}", "academic", "\n\n".join(acc)))
    for b, frac in (("1342", 0.4), ("2701", 0.4), ("145", 0.4), ("1400", 0.4), ("514", 0.4)):
        texts.append(("gutenberg", f"pg{b}", "narrative", _novel_passage(HERE / "human" / f"pg{b}.txt", frac)))
    for src, sid, reg, text in texts:
        if ("human", sid) in done:
            continue
        rec = {"arm": "human", "sample": sid, "opening": src, "register": reg, "seed": None, "text": text,
               "words": len(WORD.findall(text)), "quiz": False, "repetitive": False,
               "cuts": {str(L): cut(text, L) for L in LENGTHS}}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"human      {sid:9s} {rec['words']:4d} words", flush=True)


# ------------------------------------------------------------------ scoring
async def _score_all():
    from steer import pangram
    have = {(r["arm"], r["sample"], r["length"]) for r in load_jsonl(SCORES)}
    jobs = []
    for r in load_jsonl(SAMPLES):
        for L, text in r["cuts"].items():
            if text and (r["arm"], r["sample"], int(L)) not in have:
                jobs.append((r, int(L), text))
    print(f"{len(jobs)} Pangram calls to make", flush=True)
    sem = asyncio.Semaphore(4)
    out = open(SCORES, "a")

    async def one(r, L, text):
        async with sem:
            for attempt in range(4):
                try:
                    res = await pangram.detect(text, timeout=120)
                    break
                except Exception as ex:  # noqa: BLE001
                    if attempt == 3:
                        print(f"FAILED {r['arm']} {r['sample']} {L}: {ex}", flush=True)
                        return
                    await asyncio.sleep(3 * (attempt + 1))
            rec = {"arm": r["arm"], "sample": r["sample"], "register": r["register"], "length": L,
                   "words": len(WORD.findall(text)), "fraction_ai": res.get("fraction_ai"),
                   "fraction_ai_assisted": res.get("fraction_ai_assisted"), "fraction_human": res.get("fraction_human"),
                   "headline": res.get("headline"), "version": res.get("version"), "windows": res.get("windows")}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"{r['arm']:10s} {r['sample']:9s} {L:3d}w  AI {rec['fraction_ai']:.2f}  {rec['headline']}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))


def score():
    asyncio.run(_score_all())


# ------------------------------------------------------------------ report
def _boot(diffs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def report():
    S = {(r["arm"], r["sample"], r["length"]): r for r in load_jsonl(SCORES)}
    samples = load_jsonl(SAMPLES)
    lines = ["# Detector experiment: results", "",
             "Pre-registered in `PREREG.md`. Pangram `fraction_ai` of the **generated text alone** (no opening).", ""]
    # per-arm means
    arms = list(ARMS) + ["human"]
    lines += ["## Mean Pangram AI score by arm and length", "",
              "| arm | " + " | ".join(f"{L} words" for L in LENGTHS) + " | quiz-mode samples | repetitive samples |",
              "|---|" + "---|" * (len(LENGTHS) + 2)]
    for a in arms:
        cells = []
        for L in LENGTHS:
            v = [S[k]["fraction_ai"] for k in S if k[0] == a and k[2] == L]
            cells.append(f"{sum(v) / len(v):.2f} (n={len(v)}, {sum(x >= 0.5 for x in v)} AI-majority)" if v else "–")
        ss = [r for r in samples if r["arm"] == a]
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {sum(r['quiz'] for r in ss)}/{len(ss)} | {sum(r['repetitive'] for r in ss)}/{len(ss)} |")
    lines.append("")
    # paired comparisons
    def paired(x, y, L, reg=None):
        ids = sorted({k[1] for k in S if k[0] == x and k[2] == L})
        d = [S[(x, i, L)]["fraction_ai"] - S[(y, i, L)]["fraction_ai"] for i in ids
             if (y, i, L) in S and (reg is None or S[(x, i, L)]["register"] == reg)]
        if not d:
            return None
        m, lo, hi = _boot(d)
        return m, lo, hi, sum(v < 0 for v in d), sum(v > 0 for v in d), len(d)

    lines += ["## Paired differences (row arm minus column arm), mean [95% bootstrap CI], pairs lower / higher", ""]
    comps = [("vary_pos", "unsteered", "primary"), ("vary_pos", "constant", "primary"),
             ("vary_both", "unsteered", "secondary"), ("constant", "unsteered", "secondary"),
             ("assistant", "unsteered", "secondary")]
    lines += ["| comparison | kind | " + " | ".join(f"{L} words" for L in LENGTHS) + " |", "|---|---|" + "---|" * len(LENGTHS)]
    verdict = {}
    for x, y, kind in comps:
        cells = []
        for L in LENGTHS:
            p = paired(x, y, L)
            if p:
                m, lo, hi, nl, nh, n = p
                cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] {nl}/{nh} of {n}")
                if kind == "primary" and L in (300, 500):
                    verdict[(x, y, L)] = hi < 0
            else:
                cells.append("–")
        lines.append(f"| {x} − {y} | {kind} | " + " | ".join(cells) + " |")
    lines.append("")
    for reg in ("academic", "narrative"):
        lines += [f"### {reg} openings only", "", "| comparison | " + " | ".join(f"{L} words" for L in LENGTHS) + " |", "|---|" + "---|" * len(LENGTHS)]
        for x, y, _ in comps:
            cells = []
            for L in LENGTHS:
                p = paired(x, y, L, reg)
                cells.append(f"{p[0]:+.3f} [{p[1]:+.3f}, {p[2]:+.3f}] {p[3]}/{p[4]} of {p[5]}" if p else "–")
            lines.append(f"| {x} − {y} | " + " | ".join(cells) + " |")
        lines.append("")
    supported = verdict and all(verdict.values()) and len(verdict) == 4
    mixed = any(verdict.values()) and not supported
    lines += ["## Pre-registered verdict on H1", "",
              f"**{'SUPPORTED' if supported else 'MIXED' if mixed else 'NOT SUPPORTED'}** "
              f"(criterion: C − A and C − B both have 95% CIs below 0 at 300 and 500 words; "
              + ", ".join(f"{x}−{y}@{L}: {'yes' if v else 'no'}" for (x, y, L), v in verdict.items()) + ")", ""]
    RESULTS.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    {"gen": gen, "score": score, "report": report}[sys.argv[1]]()
