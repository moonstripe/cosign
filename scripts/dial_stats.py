"""Recompute the numbers quoted in METHODOLOGY.md from the saved generations.

    python3 scripts/dial_stats.py

Reads only files in wsp_runs/ (no model needed):
  - wsp_runs/tense-calm/compare.json   word vector vs dial vs base, strengths 4/7/10 (10 = full)
  - wsp_runs/tense-calm-v2/compare.json  the shipped tense dial (v1) vs the v2 attempt, alpha 0.5 and 1.0
  - wsp_runs/all-dials/compare.json    base vs tender / playful / lush dials, alpha 0.6 and 1.0

All counts are keyword or punctuation counts with the word lists below. They are
crude, transparent proxies; the per-run reports in wsp_runs/<axis>/report.md
hold the programmers' read-every-sample judgements.
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "wsp_runs"


def has(rx: re.Pattern, text: str) -> bool:
    return bool(rx.search(text))


def count(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text))


# ---------------------------------------------------------------- word vector vs dial (tense <-> calm)
GRAMMAR = re.compile(r"\b(tenses?|verbs?|grammar|grammatical|clauses?|past participle|present perfect|word forms?)\b", re.I)
SAYS_TENSE = re.compile(r"\btense\b", re.I)
THREAT = re.compile(r"\b(scream\w*|panic\w*|blood|gun\w*|knife|dead|death|kill\w*|danger\w*|trapped|terror\w*|frantic\w*|attack\w*|shot|police)\b", re.I)
QUIZ = re.compile(r"(how many|how much|how long|solve|step[- ]by[- ]step|express your answer|\bcalculate\b|\bA\)|\bB\)|let'?s denote|\\\()", re.I)


def word_vs_dial():
    rows = json.load(open(RUNS / "tense-calm" / "compare.json"))
    st = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for arm in ("base", "word", "dial"):
            t = r[arm]
            k = (r["strength"], arm)
            st[k]["n"] += 1
            st[k]["grammar talk"] += has(GRAMMAR, t)
            st[k]["says 'tense'"] += has(SAYS_TENSE, t)
            st[k]["threat words"] += has(THREAT, t)
            st[k]["quiz mode"] += has(QUIZ, t)
    print("## tense <-> calm: word vector vs dial (samples containing ..., out of n)")
    print("strength = signpost strength (10 = full = dial alpha 1)\n")
    cols = ["grammar talk", "says 'tense'", "threat words", "quiz mode"]
    print(f"| strength | arm  | n  | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for k in sorted(st):
        print(f"| {k[0]:g} | {k[1]} | {st[k]['n']} | " + " | ".join(str(st[k][c]) for c in cols) + " |")
    print()


# ---------------------------------------------------------------- all dials, lexical + structural
LEX = {
    "tender": re.compile(r"\b(love\w*|hug\w*|dear\w*|gentl\w*|tears?|smil\w*|hold\w*|held|warm\w*|kiss\w*|heart\w*|soft\w*|care\w*|embrace\w*|proud)\b", re.I),
    "playful": re.compile(r"\b(laugh\w*|giggl\w*|grin\w*|silly|joke\w*|funny|tease\w*|wink\w*|haha|play\w*|fun|ridiculous|mischie\w*)\b", re.I),
    "lush": re.compile(r"\b(glimmer\w*|shimmer\w*|velvet\w*|golden|amber|scent\w*|fragran\w*|glow\w*|silk\w*|bloom\w*|soft\w*|gleam\w*|lush|honey\w*|whisper\w*|dappl\w*|luminous|crimson|emerald)\b", re.I),
}
MOOD_WORD = {
    "tender": re.compile(r"\b(tender\w*|cold\w*)\b", re.I),
    "playful": re.compile(r"\b(playful\w*|grave\w*)\b", re.I),
    "lush": re.compile(r"\b(lush\w*|plain\w*)\b", re.I),
}
QEX = re.compile(r"[?!]")
CONTRACTION = re.compile(r"\b\w+'(s|re|ll|ve|d|m|t)\b|\b\w+n't\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|kind of|sort of|i guess|probably|just|actually|well,|oh|anyway)\b", re.I)
SENT = re.compile(r"[^.!?]+[.!?]")
WORD = re.compile(r"[A-Za-z']+")


def struct(texts: list[str]) -> dict:
    words = [w for t in texts for w in WORD.findall(t)]
    sents = [s for t in texts for s in SENT.findall(t)]
    n = len(texts)
    return {
        "?/! per sample": sum(count(QEX, t) for t in texts) / n,
        "contractions per sample": sum(count(CONTRACTION, t) for t in texts) / n,
        "hedges per sample": sum(count(HEDGE, t) for t in texts) / n,
        "words per sentence": float(statistics.mean(len(WORD.findall(s)) for s in sents)) if sents else 0.0,
        "mean word length": statistics.mean(len(w) for w in words) if words else 0.0,
        "quiz mode (samples)": sum(has(QUIZ, t) for t in texts),
    }


def all_dials():
    rows = json.load(open(RUNS / "all-dials" / "compare.json"))
    alphas = sorted({r["alpha"] for r in rows})
    print("## all dials vs base (12 held-out prompts x 2 seeds = 24 samples per cell)\n")
    print("Lexicon hits = total matches of the dial's own word list (see LEX in this file).\n")
    print("| dial | lexicon hits: base | " + " | ".join(f"alpha {a:g}" for a in alphas) + " | mood word said (alpha 1) |")
    print("|---|---|" + "---|" * len(alphas) + "---|")
    for d in ("tender", "playful", "lush"):
        base = sum(count(LEX[d], r["base"]) for r in rows if r["alpha"] == alphas[0])
        by_a = [sum(count(LEX[d], r[d]) for r in rows if r["alpha"] == a) for a in alphas]
        said = sum(has(MOOD_WORD[d], r[d]) for r in rows if r["alpha"] == alphas[-1])
        print(f"| {d} | {base} | " + " | ".join(map(str, by_a)) + f" | {said}/24 |")
    print()
    print("Structural features (same samples):\n")
    keys = list(struct(["x."]).keys())
    print("| setting | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    s = struct([r["base"] for r in rows if r["alpha"] == alphas[0]])
    print("| base | " + " | ".join(f"{s[k]:.2f}" if isinstance(s[k], float) else str(s[k]) for k in keys) + " |")
    for d in ("tender", "playful", "lush"):
        for a in alphas:
            s = struct([r[d] for r in rows if r["alpha"] == a])
            print(f"| {d} {a:g} | " + " | ".join(f"{s[k]:.2f}" if isinstance(s[k], float) else str(s[k]) for k in keys) + " |")
    print()


# ---------------------------------------------------------------- tense <-> calm v1 vs v2 (shipped v1)
CALM = re.compile(r"\b(peace\w*|quiet\w*|gentl\w*|soft\w*|calm\w*|serene|warm\w*|smil\w*|relax\w*|ease)\b", re.I)


def v1_vs_v2():
    rows = json.load(open(RUNS / "tense-calm-v2" / "compare.json"))
    print("## tense <-> calm: v1 (earlyW, shipped) vs v2 (tc2_final), samples containing ..., out of 12 prompts x 2 seeds\n")
    print("| alpha | arm | threat words | calm words | says 'tense' | quiz mode |")
    print("|---|---|---|---|---|---|")
    for a in sorted({r["alpha"] for r in rows}):
        for arm in ("base", "v1", "v2"):
            ts = [r[arm] for r in rows if r["alpha"] == a]
            print(f"| {a:g} | {arm} | {sum(has(THREAT, t) for t in ts)} | {sum(has(CALM, t) for t in ts)} | "
                  f"{sum(has(SAYS_TENSE, t) for t in ts)} | {sum(has(QUIZ, t) for t in ts)} |")
    print()


if __name__ == "__main__":
    word_vs_dial()
    v1_vs_v2()
    all_dials()
