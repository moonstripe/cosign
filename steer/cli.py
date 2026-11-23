"""cosign: a terminal co-writing loop with LFM3-1.7B, steered per token.

Run:  uv run cosign  [--seed 0] [--temp 0.8] [--budget 8]
"""

from __future__ import annotations

import argparse
import re
import sys

from .engine import Engine

DIM, CYAN, YELLOW, BOLD, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[1m", "\033[0m"
COLOR = {"human": "", "picked": CYAN, "model": YELLOW}

HELP = f"""{BOLD}You and the model share one document. Nothing is written unless you ask.{RESET}

  some text          you write it (a space is added at word boundaries; \\n = newline)
  <enter>            model writes the next {{budget}} tokens
  /N                 model writes N tokens            e.g. /3
  .                  model finishes the current sentence
  ?                  show the next-token candidates; then type a number to pick one
  <N                 rewind N tokens (< alone = 1)
  >tense - calm      turn a mood dial (only axes with a dial in moods/: tense-calm,
                     tender-cold, playful-grave, lush-plain)
  >tender - cold @6 for 10
                     dial at 0.6 (strength 0-10 = dial 0-1; default 4), for 10 tokens only
  >                  clear all dials
  :budget N  :temp X  :show  :save PATH  :reset  :help  :q

  {YELLOW}model{RESET}  {CYAN}picked by you{RESET}  written by you
"""

SIGNPOST_RE = re.compile(r"^>\s*(?P<c>[^@\-]+?)\s*(?:-\s*(?P<a>[^@]+?))?\s*(?:@\s*(?P<s>[\d.]+))?\s*(?:for\s+(?P<t>\d+))?\s*$")


def render(e: Engine) -> str:
    return "".join(COLOR[s] + t + RESET for s, t in e.spans())


def status(e: Engine, budget: int) -> str:
    prov = e.provenance()
    p = " ".join(f"{k} {v:.0%}" for k, v in sorted(prov.items()))
    sp = " | ".join(s.label() for s in e.signposts) or "none"
    return f"{DIM}[{len(e.ids)} tok | {p or 'empty'} | budget {budget} | temp {e.temperature:g} | signposts: {sp}]{RESET}"


def join_human(e: Engine, text: str) -> str:
    text = text.replace("\\n", "\n")
    doc = e.text()
    if doc and not doc[-1].isspace() and text and text[0].isalnum():
        text = " " + text
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    print(f"{DIM}loading {args.model}...{RESET}", file=sys.stderr)
    e = Engine(args.model, temperature=args.temp, seed=args.seed)
    budget = args.budget
    cands = []
    print(HELP.replace("{budget}", str(budget)))

    def stream(piece):
        print(YELLOW + piece + RESET, end="", flush=True)

    while True:
        print(status(e, budget))
        try:
            line = input(f"{BOLD}>>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = line.strip()

        try:
            if cmd == "":
                print(DIM + e.text()[-200:] + RESET, end="", flush=True)
                e.generate(budget, on_token=stream)
                print()
            elif re.fullmatch(r"/\d+", cmd):
                print(DIM + e.text()[-200:] + RESET, end="", flush=True)
                e.generate(int(cmd[1:]), on_token=stream)
                print()
            elif cmd == ".":
                print(DIM + e.text()[-200:] + RESET, end="", flush=True)
                e.generate(80, on_token=stream, stop=lambda p: bool(re.search(r"[.!?]\s*$|\n", p)))
                print()
            elif cmd == "?":
                cands = e.candidates(9)
                for i, c in enumerate(cands, 1):
                    bar = "#" * max(1, round(c.prob * 30))
                    shift = ""
                    if c.base_prob is not None:
                        d = c.prob - c.base_prob
                        shift = f"  {DIM}(unsteered {c.base_prob:.0%}, {'+' if d >= 0 else ''}{d:.0%}){RESET}"
                    print(f"  {i}  {c.text!r:<16} {c.prob:6.1%} {bar}{shift}")
            elif cands and re.fullmatch(r"[1-9]", cmd) and int(cmd) <= len(cands):
                c = cands[int(cmd) - 1]
                e.pick(c.token_id)
                cands = []
                print(render(e)[-400:])
            elif re.fullmatch(r"<\d*", cmd):
                e.rewind(int(cmd[1:] or 1))
                print(render(e)[-400:])
            elif cmd == ">":
                e.clear_signposts()
            elif cmd.startswith(">"):
                m = SIGNPOST_RE.match(cmd)
                if not m:
                    print("could not parse signpost, try:  >ocean @6 for 10")
                    continue
                e.add_signpost(
                    m["c"].strip(),
                    m["a"].strip() if m["a"] else None,
                    float(m["s"]) if m["s"] else 4.0,
                    int(m["t"]) if m["t"] else None,
                )
            elif cmd.startswith(":"):
                parts = cmd[1:].split(maxsplit=1)
                op, arg = parts[0], (parts[1] if len(parts) > 1 else "")
                if op in ("q", "quit", "exit"):
                    break
                elif op == "budget":
                    budget = int(arg)
                elif op == "temp":
                    e.temperature = float(arg)
                elif op == "show":
                    print(render(e))
                    for k, v in sorted(e.provenance().items()):
                        print(f"  {COLOR[k]}{k:<7}{RESET} {v:.0%}")
                elif op == "save":
                    with open(arg or "draft.txt", "w") as f:
                        f.write(e.text())
                    print(f"saved {arg or 'draft.txt'}")
                elif op == "reset":
                    e.reset()
                    e.clear_signposts()
                elif op == "help":
                    print(HELP.replace("{budget}", str(budget)))
                else:
                    print("unknown command, :help")
            else:
                e.write(join_human(e, line))
                cands = []
        except Exception as ex:  # keep the session alive
            print(f"error: {ex}")


if __name__ == "__main__":
    main()
