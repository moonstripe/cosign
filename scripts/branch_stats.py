"""Read the 48% branch straight out of a Cosign save file.

    python3 scripts/branch_stats.py [save.cosign.json] [node-id]

Defaults to samples/ai_scientist_branch_48pct.cosign.json and the node the
save was left on. Prints what the branch is made of (who wrote what, which
moods were dialled, how hard) and the Pangram AI score after each decision.
"""

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "samples" / "ai_scientist_branch_48pct.cosign.json"
doc = json.load(open(path))
nodes = doc["tree"]["nodes"]
target = sys.argv[2] if len(sys.argv) > 2 else doc["tree"]["cur"]

chain = []
i = target
while i:
    chain.append(nodes[i])
    i = nodes[i]["parent"]
chain.reverse()

imported = [n for n in chain if n["via"] == "import"]
decisions = [n for n in chain if n["via"] not in ("root", "import")]
kinds = Counter(n["via"] for n in decisions)
writes = [n for n in decisions if n["via"] == "opt-tab"]
steered = [n for n in writes if n["meta"].get("steer")]
moods = Counter(s["toward"] for n in steered for s in n["meta"]["steer"])
strengths = [s["strength"] for n in steered for s in n["meta"]["steer"]]

chars = Counter()
for src, text in re.findall(r'data-src="(\w+)"[^>]*>(.*?)</span>', nodes[target]["html"], re.S):
    chars[src] += len(html.unescape(re.sub(r"<[^>]+>", "", text)))
total = sum(chars.values())

print(f"file      {path.name}")
print(f"node      {target}  ({len(nodes[target]['text'].split())} words)")
print(f"start     {len(imported)} imported chunk(s); chunk score {imported[-1]['pg']['ai']:.0%} AI" if imported and imported[-1].get("pg") else "start     -")
print(f"decisions {len(decisions)}: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
print(f"model writes (opt-tab) {len(writes)}, steered with the wheel {len(steered)}")
print(f"moods     " + ", ".join(f"{m} {c}" for m, c in moods.most_common()))
if strengths:
    print(f"strength  min {min(strengths):.2f}  median {sorted(strengths)[len(strengths) // 2]:.2f}  max {max(strengths):.2f}  (redline 0.65)")
print("authorship " + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in chars.most_common()))
scores = [n["pg"]["ai"] for n in decisions if n.get("pg")]
if scores:
    print(f"Pangram AI after each decision: first {scores[0]:.2f} -> last {scores[-1]:.2f} ({len(scores)} scored)")
    print("  " + " ".join(f"{s:.2f}" for s in scores))
elif doc.get("pangram"):
    print(f"Pangram AI for the whole document: {doc['pangram']['fraction_ai']:.0%} ({doc['pangram'].get('headline', '')})")
