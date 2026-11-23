# tender ↔ cold: working notes

## Step 1: get contrast text from the model itself (passes 1–2)
- I wrote 16 neutral narrative openings (a letter, a phone call at midnight, a hospital corridor, a nurse, a dog, a funeral, and so on).
- Pass 1 put a raw-text header in front of each one: "From a novel. The scene is unbearably tender, warm and loving:\n\n<opening>". I took 8 samples per opening, 64 tokens each, temperature 0.8, seed 0.
- Pass 2 used the same openings and settings with a matching cold header: "...The scene is detached, clinical and emotionally distant:".
- Sometimes the model wandered back into the header or talked about writing ("The scene is unbearably tender...", "We want to express emotion in the text..."). Where that happened I cut the sample at that point and kept only the story text before it.

## Step 2: filter (done locally, before anything was computed)
- I cut each sample at the first meta or framing marker (the header phrase, "from the same passage", "we need to", "[Name]", quiz cues and so on), then trimmed it back to the last full sentence. 20 samples were cut this way.
- I dropped samples that contained a tender or cold stem (13), samples in maths or quiz mode (1), and samples that ended up shorter than 20 words or broken (15).
- I paired tender sample i with cold sample i for the same opening, keeping min(#tender, #cold) per opening.
- **Survivors: 106 matched pairs (212 texts) out of 128 + 128 generated.** Every opening kept between 4 and 8 pairs.
- Reading a random six pairs: the tender side has endearments ("dear boy", "My baby"), holding, trembling voices and love. The cold side has staring blankly, a barista who never speaks, walking without speaking, flat logistics. The register really does differ, but the signal is noisy because some cold samples are just sad or neutral.

## Step 3: read the samples with the framing stripped (pass 3)
- Each surviving continuation was teacher-forced after the bare opening only, with no header. I recorded the mean residual at every layer and the w2 inputs at layers 7, 9, 11, 13, 15 and 17.
- The paired difference (tender − cold) is consistent at every layer: about 0.89 of pairs project positively onto the mean difference, with d' ≈ 1.1–1.2. The difference is 7–16% of the residual norm. These figures are in-sample.

## Step 4: build the edit (earlyW recipe)
- Rank-one edits on w2 at layers 7, 9 and 11 (c = 0.3) and 13, 15 and 17 (c = 0.1).
- u = the mean paired residual difference at that layer. A = c·|ȳ|·normalize(post_ffn_norm.w ⊙ u), where |ȳ| is the mean norm of the w2 output. B = m/|m|², where m is the mean w2 input, so x·B ≈ 1.

## Step 5: vocabulary check, and the fix it forced
- On the raw directions, " tender" came out at rank 5 (L17), 17 (L13), 32 (L15) and 585 (L9), and rank 8 for the summed direction. That fails the top-2,000 rule. This happened even though no training text contained the word: the model's own "tender register" direction points at the word.
- Fix (closed form): in each layer's A, I removed the component along g ⊙ g_final ⊙ E[t] for the first tokens of the mood words (" tender", " tend", " cold", " Cold"), then rescaled back to the original norm. This sets the direct logit contribution to those tokens to zero.
- After the fix, every mood-name token ranks between 13,710 (" Tender", via the " T" first token at L11) and about 126k, in every layer. The summed direction's top tokens are still register words: expression, hug, tears, forehead, love, affection, warmth, blessings.
- Saved `tc_v1` and `tc_rand` (a norm-matched random u at each layer, put through the same projection and scaling).

## Step 6: test generations (passes 4–6)
- Pass 4 used 8 openings (one, the letter, overlaps the elicitation set), comparing off, v1@0.5, v1@1 and rand@1. v1@1 is clearly warmer (hugs, "I love you", tears). The random control looks like baseline.
- Pass 5 used 8 unseen openings, comparing off, v1@1, v1×1.5 and rand×1.5. v1×1.5 is over the top and breaks more often (loops, pronoun confusion, meta "this is a touching scene", one drift into Chinese). My tally for v1@1 vs off was 40 tenderer / 7 broken / 17 unchanged.
- To be conservative I set the final α=1 to 0.8× v1 (`tc_final`), with a matching control `tc_rand_final`.
- Pass 6 used 8 new unseen openings, comparing off, final@0.5, final@1 and rand_final@1. My tally for final@1 vs off was 34 tenderer / 11 broken / 19 unchanged; the baseline itself had about 13/64 broken or quiz on these prompts.
- Mood words: "tender*" never rose. "cold" rose only on the snow opening, as literal temperature, and the random control matched it (7 vs 8).

## Step 7: submit
- `submit_dial('tc_final')` wrote /data/moods/tender__cold.pt, which I checked by loading it. The lab then crashed while recording the submission (a relative_to path bug), so `status` shows `submitted: []`. I did not edit lab code.
