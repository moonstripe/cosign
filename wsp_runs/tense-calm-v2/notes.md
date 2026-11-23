# Notes: tense ↔ calm dial v2 (LFM3-1.7B)

## Step 1 — elicit contrast data from the model
- 16 neutral narrative openings (work/openings.json). No mood words, no numbers/countdowns.
- Framings (raw-text genre headers, deliberately without the words tense/calm so the model is not primed to echo them):
  - tense: "From a suspense thriller. Fear, danger, dread, panic.\n\n<opening>"
  - calm:  "From a gentle slice-of-life novel. Peace, ease, warmth, quiet.\n\n<opening>"
- 12 samples per prompt, 72 tokens, T=0.8, unedited model. Two generate passes (8 openings each).
- Operational mishap: I launched the two elicitation passes concurrently. The lab recorded them both as pass 1 and the second overwrote the first, so only the 8 openings of the second half (seed 1) survive (gens(1)). Only 1 pass was charged. Given the 45-min wall-clock limit (each generate pass ≈8 min on the shared GPU) I did not re-run the first half: the contrast set uses **8 openings, not 16**. Lesson: never run two `wsp forward` calls at once.

## Step 2 — filter
- Dropped samples with mood stems (tense/tension/tenses/calm*), quiz/maths cues, framing echoes (thriller/novel/"From a"), or broken text (<25 words, low distinct 4-gram ratio, repeated substrings).
- Survival: tense 88/96, calm 78/96 (calm samples mostly dropped for literally saying "calm": 17; quiz 1 each; mood 7 tense). Matched by opening -> 75 pairs (8–11 per opening).
- Read a dozen by eye: tense side has gunshots, sirens, "heart pounded", ominous wind; calm side has lilac scent, golden haze, lavender. Register really differs.

## Step 3 — read teacher-forced after the bare opening (pass 2)
- 150 items (75 tense + 75 calm continuations), prefix = bare opening only. Captured residual means and w2 inputs at layers 7–17 odd.
- u_L = mean residual(tense) − mean residual(calm) at layer L output.

## Step 4 — the edit (earlyW recipe, reused)
- Rank-1 on feed_forward.w2 at L = 7,9,11 (c=0.3) and 13,15,17 (c=0.1): A = c·|h̄_L|·normalize(post_ffn_norm.weight ⊙ u_L), B = h̄/|h̄|² where h̄ = mean w2 input (x·B = 1.00 ± 0.05–0.08).
- Candidate `tc2_a`; control `tc2_rand` = same B, A replaced by a random vector of identical norm per layer.

## Step 5 — vocabulary check (direction post_ffn_norm.w ⊙ A, through final norm & tied embedding)
- Early layers (7/9/11) boost threat/accusation words (blamed, accused, armed, confrontation, alarming, threatening, traps, malicious) and suppress warmth/sunshine/leisure/softly/gently/relaxed.
- Late layers (15/17) boost frightening, nightmare, scary, deadly, ransom, police, emergency, scream, panic; suppress relax, sunlight, breeze, leisure, enjoying.
- No countdown/number tokens in any top 25 (the v1 ' ten' leak is gone).
- Mood-name tokens: all first-token ranks are far outside the top 2,000 (best: ' tense' 35,154 at L17; ' Calm' first piece 29,290 at L7). Only sub-word continuation pieces get closer (e.g. the 2nd piece of ' Calm' rank 1,046 at L11) — these are generic fragments, not the mood word.

## Step 6 — test pass (pass 3): 16 held-out openings × 4 samples, 64 tokens, seed 7, settings off / tc2_a α=.5 / tc2_a α=1 / random α=1
- Automatic counts (64 samples each): mood words 0/0/0/0; tense-lexicon hits 4/13/19/5; calm-lexicon hits 16/7/1/19; repetition-flagged 1/4/4/1; quiz-cue 16/12/11/10.
- Read flagged samples: at α=1 all 4 are real counting/looping repeats (6%); at α=.5, 2 are false positives (water, brothers) and 2 are real loops (Sam, Peter "knocks 100 times") ≈3%.
- Read sample 0 of every opening off vs α=1: tenser-and-fluent 6/16 (museum robbed, brothers fight + police, librarian "demanded", packing → murder trial, night shift "in danger", grandma's "curse"); broken/looping 3/16 (mountain hut, bakery, Peter) + 1 circular (captain); unchanged 6/16 (several still quiz mode).
- Decision: α=1 of tc2_a exceeds ~5% broken, so the submitted dial `tc2_final` = tc2_a × 0.5 (its α=1 is the tested α=.5 point). Submitted.
