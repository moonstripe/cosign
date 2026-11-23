# Report: tender ↔ cold dial for LFM3-1.7B (run `tender-cold`)

**Final dial: `tc_final`** (concept "tender", anti "cold"). 6 of 15 passes used.

`submit_dial('tc_final')` wrote `/data/moods/tender__cold.pt`, and the file checks out (concept, anti, deltas, meta name `tc_final`). The lab then raised `ValueError: '/data/moods/tender__cold.pt' is not in the subpath of '/root'` in its bookkeeping, so `wsp status` still shows `submitted: []`. This is a lab path bug. I did not touch lab code.

## What I changed
- Rank-one edits on `feed_forward.w2` at layers 7, 9, 11, 13, 15 and 17: W' = W + α·A·Bᵀ.
- **u_L** is the mean paired (tender − cold) residual at the output of layer L, from teacher-forced reads with the framing stripped.
- **A_L** = c_L·|ȳ_L|·normalize(post_ffn_norm.w ⊙ u_L).
  - c = 0.24 at layers 7, 9 and 11, and 0.08 at 13, 15 and 17. This is the earlyW recipe (0.3/0.1) scaled by 0.8.
  - |ȳ_L| is the mean norm of the w2 output over the recorded items.
  - The component of A along post_ffn_norm.w ⊙ final_norm.w ⊙ E[t] is projected out for t ∈ {" tender", " tend", " cold", " Cold"}, then A is rescaled back to its original norm. This is closed form: one QR per layer.
- **B_L** = m/|m|², where m is the mean w2 input, so x·B ≈ 1 on typical text.
- No gradients and no fitting. All of this is means, norms and one projection.

## Contrast data
- 16 neutral narrative openings × 8 samples, from two raw-text headers: "From a novel. The scene is unbearably tender, warm and loving:" and "...detached, clinical and emotionally distant:". 64 tokens, temperature 0.8.
- **Filtering** (256 samples in total):
  - 20 were cut at a meta or framing echo (the header repeated, "we want to express emotion in the text", "[Name]", and so on), keeping the story text before the cut.
  - 13 dropped for a tender/cold stem.
  - 1 dropped for maths.
  - 15 dropped as too short or broken (under 20 words).
- Pairs were matched by opening: **106 matched pairs (212 texts) survived**, 4 to 8 per opening.
- I read a sample of the pairs. Tender texts have endearments, holding, tears and "I love you". Cold texts have blank stares, people who don't speak, and logistics. The register difference is real but noisy: some cold samples are just sad or neutral.
- The continuations were teacher-forced after the **bare opening** (pass 3). The paired difference is very consistent: about 0.89 of pairs lie on the positive side of the mean difference at every layer, with d' ≈ 1.1 (in-sample).

## Vocabulary check (final norm weight × tied embedding, all 128k tokens)
- **Before the fix, the check failed.** " tender" ranked 5 (L17), 17 (L13), 32 (L15) and 585 (L9), and 8 for the summed edit. The data contained no "tender" at all, so the model's own tender-register direction is aligned with the word itself.
- **After projection** (the submitted dial; scaling does not change ranks):
  - Mood-token ranks per layer, minimum over all forms: L7 73,057; L9 74,097; L11 13,710; L13 73,122; L15 67,322; L17 76,095.
  - The minimum at L11 is " Tender" via the " T" first token. Every other form ranks about 67k–126k.
  - Multi-token forms (tenderly, tenderness, coldly, coldness and the capitals) were ranked by their first token.
  - **No mood token is in the top 2,000** for any layer or for the sum.
- Summed edit after projection:
  - Top 25 boosted: expression, hug, sinc, Bundled, tears, expression, expressing, 사랑 ("love"), forehead, Expression, _expression, love, Expression, Heart, UIT, affection, capturing, expressions, warmth, 愛 ("love"), 857, Express, remembering, express, blessings.
  - Top 12 suppressed: isolated, landsc, 序, grounds, ores, ost, ger, buz, dull, sk, furnished, interesting.
  - Per layer: L15 and L17 boost sweet, protective, affection, hug, kiss, cher, warmth. L7, L9 and L11 boost mostly junk or subword tokens; those layers act through later computation, not directly on the logits.

## Dose–response
Paired samples (same seed). "warm-lexicon" is an automatic count of words like love/hug/dear/gentle/tears/smile/hold over 64 samples. Real judgement comes from reading (below).

| pass | setting | final-dial equivalent α | warm-lexicon | stem tender/cold |
|---|---|---|---|---|
| 4 | off | 0 | 36 | 0 |
| 4 | tc_v1 α0.5 | 0.625 | 76 | 1 (tenderness) |
| 4 | tc_v1 α1 | 1.25 | 168 | 0 |
| 5 | off | 0 | 14 | 1 |
| 5 | tc_v1 α1 | 1.25 | 165 | 0 |
| 5 | tc_v15 α1 | 1.875 | 203 | 0 |
| 6 | off | 0 | 16 | 3 |
| 6 | **tc_final α0.5** | 0.5 | 56 | 4 |
| 6 | **tc_final α1** | 1.0 | 143 | 8 |

From reading:
- α≈0.5: subtle. Some samples gain a hug or "I'm so proud of you".
- α=1: clearly tender in about half of samples.
- α≈1.25: strong but sometimes saccharine.
- α≈1.9 (v15): over-the-top ("You are my everything"), with more pronoun confusion, loops, meta-commentary ("This is a touching scene") and one drift into Chinese. That is why α=1 was set at 0.8× v1.

## Norm-matched random control
A random u per layer with the same per-layer norm, the same mood-token projection and the same B.

| setting | warm-lexicon |
|---|---|
| tc_rand α1 (pass 4) | 28, vs 36 off |
| tc_rand15 (pass 5) | 22, vs 14 off |
| tc_rand_final α1 (pass 6) | 30, vs 16 off |

Reading those samples, the random control looks the same as baseline in register. Its quiz/maths drift is about the same as the real dial's (pass 4 regex: off 6, rand 12, v1@1 13; pass 6: off 11, rand 5, final 4). So much of the quiz drift seen at α=1 comes from perturbation size, not the direction.

## Held-out paired comparison (read by me, dial vs off, same seed)
Not tenderer, broken and unchanged are judged against the same-seed off sample.

- **tc_final α=1, pass 6 (8 unseen openings × 8 = 64 pairs):**
  - **34 tenderer and fluent**
  - **11 broken** (repetition loops, pronoun confusion, quiz or translation mode)
  - **19 unchanged or not tenderer**
- The baseline (off) samples in the same pass had about 13/64 broken or quiz, so most of those 11 occur on prompts that are already quiz-prone. The kitten and snow openings went to quiz or loops in about half of samples even with the dial off.
- Same-seed agreement on breakage is only partial.
- Per opening, the strongest effects were grandma pouring tea (7/8), son's first day (7/8), kitten, soldier, diner and ferry (4/8 each). The snow/village opening showed 0/8.
- **tc_v1 α=1 (≈ final α=1.25), pass 5 (64 pairs): 40 tenderer and fluent / 7 broken / 17 unchanged** (baseline about 6/64 broken). The grandfather opening was already warm at baseline (2/8 changed).
- **Mood words, dial on vs off (pass 6, final): "tender\*" 0 vs 0. "cold\*" 8 vs 3.**
  - 7 of the 8 "cold" hits come from the "Snow covered the whole village" opening, as literal temperature ("It was very cold that day").
  - The norm-matched random control gives 7 on the same pass, so this is temperature talk on a snow prompt, not the dial leaking its word.
  - Across passes 4–6, "tender\*" appeared once in total with any dial on (v1 α0.5) and 0 times at full strength.

## Honest limits
- **Breakage:** at α=1 about 11/64 held-out samples are broken. That is above the ≤5% target in absolute terms, but similar to this base model's own quiz/loop rate on the same prompts (about 13/64). I cannot claim the dial adds less than about 5% breakage with 64 pairs. It seems to add few new failures, but the evidence is noisy.
- **Sentimentality:** "tender" at high α turns into saccharine "I love you" and "my baby" talk. It sometimes inserts affection that makes no narrative sense (a landlord saying "Sweetie", a doctor saying "most beautiful girl"), and pronoun and role confusion rises with α. The tender side is easy to see; I did not test the cold end (α can't go negative), so this is really a "more tender" dial.
- **Prompt dependence:** the effect is weak on openings that are already warm or that the base model treats as puzzles (kitten, waitress, snow).
- **How the projection works:** it only zeroes the *direct* logit effect of the mood tokens through the residual write. Layers 7–11 act through downstream computation, which this linear check does not see. The generation counts are the real evidence, and they show no rise in "tender".
- **Evaluation size:** the warm-lexicon counts are crude, and the tallies are one reader's judgement on 64 pairs per pass.
- **Lab bug:** `submit_dial` wrote the dial file but crashed while recording it, so the submission list is empty.
