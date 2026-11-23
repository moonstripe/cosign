"""Fill steer/wsp/tasks/TEMPLATE.md for one mood axis.

  uv run python -m steer.wsp.make_task tense calm --run tense-calm-v2 --budget 15
"""
import argparse
from pathlib import Path

TASKS = Path(__file__).parent / "tasks"

DESC = {
    "tense": "anxious, clipped, suspenseful, heightened stakes", "calm": "unhurried, settled, peaceful, low stakes",
    "tender": "warm, affectionate, intimate, gentle care", "cold": "detached, clinical, emotionally distant",
    "playful": "light, witty, teasing, buoyant", "grave": "solemn, weighty, serious, mournful",
    "lush": "rich, sensory, ornate, image-dense", "plain": "spare, literal, unadorned, matter-of-fact",
}
STEMS = {
    "tense": ["tense", "tension", "tenses"], "calm": ["calm", "calmly", "calmness"],
    "tender": ["tender", "tenderly", "tenderness"], "cold": ["cold", "coldly", "coldness"],
    "playful": ["playful", "play", "playfully"], "grave": ["grave", "gravely", "gravity"],
    "lush": ["lush", "lushly"], "plain": ["plain", "plainly"],
}


def make(concept, anti, run, budget, lab_url="http://127.0.0.1:8790"):
    stems = STEMS.get(concept, [concept]) + STEMS.get(anti, [anti])
    text = (TASKS / "TEMPLATE.md").read_text().format(
        concept=concept, anti=anti, run=run, budget=budget, lab_url=lab_url,
        concept_desc=DESC.get(concept, concept), anti_desc=DESC.get(anti, anti),
        forbidden=", ".join(stems),
        forbidden_tokens=", ".join(f'" {s}"' for s in stems) + ", and the capitalised forms",
    )
    p = TASKS / f"{run}.md"
    p.write_text(text)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("concept"); ap.add_argument("anti")
    ap.add_argument("--run"); ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--lab-url", default="http://127.0.0.1:8790")
    a = ap.parse_args()
    print(make(a.concept, a.anti, a.run or f"{a.concept}-{a.anti}", a.budget, a.lab_url))
