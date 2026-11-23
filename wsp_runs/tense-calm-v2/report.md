# Report: tense ↔ calm dial v2 for LFM3-1.7B (submitted: `tc2_final`)

## What I changed
Rank-1 edits on `feed_forward.w2` at layers 7, 9, 11, 13, 15, 17 (6 modules, rank 1 each). This is the earlyW recipe: A_L = s·c_L·|h̄_L|·normalize(post_ffn_norm.weight_L ⊙ u_L), B_L = x̄/|x̄|² (mean w2 input, so x·B = 1.00 ± 0.05–0.08). c = 0.3 at 7/9/11 and 0.1 at 13/15/17. u_L is the mean decoder-layer-L residual on tense continuations minus calm continuations. The submitted dial uses s = 0.5: its α=1 equals α=0.5 of the full-strength candidate `tc2_a`, because `tc2_a` at α=1 broke 6% of samples. Everything is closed-form, with no gradients.

## Contrast data (elicited from the model)
- Framings were raw-text headers that do not contain the mood words. Tense: "From a suspense thriller. Fear, danger, dread, panic.\n\n<opening>". Calm: "From a gentle slice-of-life novel. Peace, ease, warmth, quiet.\n\n<opening>". 12 samples per prompt, 72 tokens, T=0.8.
- **Only 8 openings survived, not the planned 16.** I ran two elicitation passes at the same time, and the lab stored both as pass 1, so the first half was overwritten. Only one pass was charged. Because of the time limit I did not re-run it. This is below the spec's "a dozen or more".
- Filters: mood stems (tense/tension/tenses/calm*), quiz or maths cues, framing echoes, and broken text (short, repetitive). **Survival: tense 88/96, calm 78/96.** 17 calm samples were dropped for literally saying "calm". Matching by opening gave **75 pairs** (7–11 per opening).
- I read the samples myself and the registers clearly differ: gunshots, sirens and a pounding heart on one side, lilac, golden haze and lavender on the other.
- Teacher-forced `read` used the **bare opening** as the prefix (150 items, pass 2).

## Vocabulary check (post_ffn_norm.w ⊙ A → final norm → tied embedding, 128k tokens)
- L7 top: blamed, res, armed, rej, discovers, accused, resolved, …, confrontation, alarming, urg. Bottom: mild, warmth, sunshine, leisure, warm, softly, gently.
- L9 top: negatives, -force, blame, liability, suspects, allegation, unknowns. Bottom: warm, sunshine, relaxed.
- L11 top: malicious, subjected, traps, threatening, unclear, -force. Bottom: relaxed, relaxation, relax, evening.
- L13 top: malicious, drastic, severely, aggressive, bankrupt, litigation. Bottom: relaxed, smooth, porch, leisure, relaxation.
- L15 top: drilling, ransom, alarming, frightening, suspicious, nightmare, scary, deadly, armed, lockdown. Bottom: relaxation, relax, sunlight, warm, relaxing, leisure.
- L17 top: drilling, police, emergency, nightmare, scream, ransom, frightening, shocking, panic, armed. Bottom: birds, breeze, enjoying, sunlight, leisure, comfortably, aroma.
- (The full top-25 and bottom-12 lists for each layer are in the kernel output, summarised in notes.md.) There are **no countdown or number tokens**, so the v1 ' ten' leak is gone. Some junk sub-word pieces appear ('/div', 'drill').
- **Mood-name first-token ranks** (best across layers): ' tense' 35,154 (L17; 47k–127k elsewhere); ' tension' ≥ 98,753; ' calm' ≥ 40,329; ' Tense' ≥ 49,138; ' Calm' ≥ 29,290. ' tenses', ' calmly', ' calmness' and the capitalised forms split into several tokens, and their first pieces rank the same way. **None are in the top 2,000.** Only generic second sub-word pieces come close: the second piece of ' Calm' ranks 1,046 at L11, and the second piece of ' Calmness' ranks 2,402 at L9.

## Dose–response, control and held-out (pass 3)
Setup: 16 held-out openings (not used in elicitation) × 4 paired samples, 64 tokens. Settings are tc2_a at α = 0, 0.5 and 1 (= tc2_final at α 0, 1 and 2), plus the random control.

| setting | tense-lexicon hits | calm-lexicon hits | mood words | looping/broken (read) | quiz-cue |
|---|---|---|---|---|---|
| off | 4 | 16 | 0 | 1/64 | 16 |
| tc2_final α=1 (tc2_a 0.5) | 13 | 7 | 0 | ~2/64 (2 of 4 flagged were false positives) | 12 |
| tc2_a α=1 (2× submitted) | 19 | 1 | 0 | 4/64 | 11 |
| random, norm-matched, α=1 of tc2_a | 5 | 19 | 0 | 1/64 | 10 |

- The random control does nothing measurable, which suggests the effect comes from the direction and not from the size of the perturbation.
- Paired reading, sample 0 of each opening, off vs tc2_a α=1: **6/16 tenser and fluent** (museum robbed; brothers fight and a police inspector appears; librarian "demanded"; a note turns into a murder trial; night shift "in danger"; grandma's "curse"), **3/16 looping** plus 1 circular, and **6/16 unchanged**. I did not do a full paired read at the submitted strength. Its lexicon shift is about half as large.
- **Mood words with the dial on vs off: 0 vs 0** at every setting.

## Limits (honest)
- The effect is moderate, not dramatic. The model mostly adds plot threats (crime, danger, conflict) rather than clipped, anxious prose rhythm, and about a third of samples barely change, especially in quiz mode.
- The base model goes into quiz or word-problem mode about 25% of the time. The dial reduces this slightly (16 → 11–12) but does not fix it.
- The contrast set has 8 openings and 75 pairs, below the dozen-plus openings the spec asks for, because of my concurrency mistake.
- At the submitted α=1, the broken rate is about 3% (2 real loops out of 64). That comes from a single 64-sample pass, so the uncertainty is wide. The full-strength candidate broke 6%, so I halved it.
- The tense/calm "lexicon" counts are crude keyword proxies. The judgements of register and fluency come from my own reading of a subset, not a blind rating.
- The budget was not the limit (3 of 15 passes used). Wall-clock time was: each generate pass took about 8.5 minutes on the shared GPU.
