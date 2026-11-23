# lush ↔ plain — working notes

## Step 1: elicit contrast data (passes 1–2)
- I used 16 neutral narrative openings, e.g. "She opened the letter at the kitchen table and".
- Each opening got two raw-text headers:
  - Lush: "From a novel. The prose here is lush, ornate and sensory, every sentence dense with rich imagery:\n\n<opening>".
  - Plain: "From a novel. The prose here is plain, spare and literal, every sentence short and matter-of-fact:\n\n<opening>".
- Sampling: 16 samples per prompt, 64 tokens, T=0.8. Pass 1 covered 8 openings and pass 2 the other 8, for 256 lush and 256 plain samples.
- Note: a failed (empty-stdin) forward call happened once by accident. It was free and was not recorded as a pass.

## Step 2: filter
- Dropped samples containing lush/lushly/lushness/plain/plainly/plainness: 4 plain samples, 0 lush.
- Dropped meta or quiz samples, where the text talks about the prose, author, task, sentence, imagery or "matter-of-fact", or contains a Question/Answer, "=", A)/B), or a templated [Name]: 21 plain and 5 lush. The plain header made the model comment on the style more often ("— The author is not evident from this excerpt", "The task is to select…").
- No samples were dropped as short, repetitive or arithmetic.
- Survivors: 251/256 lush and 231/256 plain. I kept 8 matched pairs per opening, which gives **128 pairs (256 items)**, all matched by opening.
- Register check on the kept samples:
  - Mean word length: lush 4.44 vs plain 4.07.
  - Commas per 100 words: lush 7.6 vs plain 5.1.
  - "like"/"as if" per sample: lush 0.52 vs plain 0.08.
  - Reading 5 random pairs confirmed the difference. Lush samples read like "the evening air seemed to kiss his skin … a gentle, liquid embrace". Plain samples read like "went to the house of his sister. The rain had been falling all day." Some plain samples are only mildly plainer, and a few still have a figurative line.

## Step 3: read teacher-forced (pass 3)
- Each kept continuation was read after the bare opening only, with no header. I recorded residual means for all layers and w2 inputs for layers 7/9/11/13/15/17.
- The lush−plain mean difference is 17–34% of the residual norm. The in-sample sign agreement of per-pair projections is 124–128/128 at every layer.

## Step 4: the edit (earlyW recipe, reused)
- Rank-1 edit on feed_forward.w2 at layers 7/9/11 (c=0.3) and 13/15/17 (c=0.1).
- A = c·|h̄|·normalize(post_ffn_norm.w ⊙ u_L), where u_L is the lush−plain residual mean at layer L and |h̄| is the mean norm of the pre-norm MLP output.
- B = x̄/|x̄|², the mean w2 input. The gate x·B is 1.00 ± 0.04–0.09.
- Relative edit size ‖ABᵀ‖/‖W‖: 1.2%, 0.7% and 1.1% at layers 7/9/11, and 0.26–0.33% at layers 13/15/17.
- Candidate `lp_a`. Control `lp_rand` uses the same B and the same per-layer |A|, with a random unit direction.

## Step 5: vocabulary check (g ⊙ A → final-norm weight → tied embedding, all 128k tokens)
- Combined top tokens: echo, radi(ant), backdrop, puls(e), sap, freshly, fragr(ance), tendr(ils), pale, azure, dew, dissol(ve), winds, illum, palette, lum(inous).
- Combined bottom tokens: promptly, quit, remar(k), stairs, abrupt, realized, noticed, trash, defec.
- Layers 15/17 are the most "sensory": fragr, tendr, ozone, azure, dew, clouds, casc(ade), violet, breeze. They suppress lunch, parking, bus, dinner, packing, promptly.
- Layers 7/9/11 are noisier, mixing sunlight, fluorescence and melt with subword junk.
- Mood tokens:
  - " plain", " Plain", "plain" and "Plain" are near the bottom (rank 36k–128k), so they are suppressed rather than boosted.
  - " lush" is not a single token (" l"+"ush"). "ush" ranks ≥3,460 everywhere. The generic " l" token ranks 1,091 at L15, 1,674 at L17 and 3,001 combined.
  - Single-token "lush" (no space) ranks 24k–105k.
  - The suffix "ly" ranks 1,431–7,561.
  - No whole mood-name token is in the top 2,000. The only sub-2,000 pieces are generic subwords (" l", "ly"), which are not mood names on their own.

## Step 6: first held-out test (pass 4)
- Setup: 8 new openings × 8 samples, comparing off, lp_a α=0.5, lp_a α=1 and lp_rand α=1.
- lp_a α=1 was strongly lush ("the tide was drawing back in slow, silver fingers").
  - It also pulled many samples out of the base model's quiz/word-problem mode.
  - It broke 4/64 samples (6%): one rhyme gibberish, one "church church church" loop, one repetition loop and one "Moral:" quiz drift.
- lp_a α=0.5 was lusher in about 24/64 samples.
- lp_rand produced no lusher samples. 2–3 were odd, similar to off.
- Mood words: 0 in all four settings.
- Decision: rescale to `lp_b` = 0.85·lp_a so that α=1 sits a little below the point where breakage started.

## Step 7: second held-out test on fresh openings (pass 5)
- Setup: off, lp_b α=0.5, lp_b α=1 and lp_rand_b α=1. Results are in report.md.

## Step 8: submit
- Called submit_dial('lp_b'). The lab wrote /data/moods/lush__plain.pt, and that file's meta name is lp_b.
- The call then raised a ValueError in the lab's own bookkeeping (path.relative_to(ROOT)), so `status.submitted` is still empty. This is a lab bug. I did not retry or work around it.
