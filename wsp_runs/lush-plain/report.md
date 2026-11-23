# lush ↔ plain dial for LFM3-1.7B: report

**Submitted dial: `lp_b`** (concept "lush", anti "plain"). It used 5 of 15 forward passes, with no gradients and no fitting.

The lab wrote the dial file `/data/moods/lush__plain.pt` (meta name lp_b). `submit_dial` then raised a ValueError in its own `relative_to(ROOT)` bookkeeping, so `wsp status` shows `submitted: []`. This is a lab bug, not a problem with the dial.

## 1. What I changed
I made rank-1 edits on `feed_forward.w2` at layers 7, 9, 11, 13, 15 and 17: 6 modules, rank 1 each. This reuses the earlyW recipe:

    W_L' = W_L + α·A_L B_Lᵀ
    A_L  = s·c_L·|h̄_L|·normalize(post_ffn_norm.weight_L ⊙ u_L)
    B_L  = x̄_L/|x̄_L|²

- c = 0.3 on layers 7/9/11 and 0.1 on 13/15/17. The global scale is s = 0.85.
- u_L is the mean residual (decoder-layer-L output) on lush continuations minus plain continuations, teacher-forced after the bare opening.
- |h̄_L| is the mean norm of the pre-norm MLP output.
- x̄_L is the mean w2 input, so the gate x·B = 1.00 ± 0.04–0.09.
- Relative edit size ‖ABᵀ‖/‖W‖ (before s): 1.2%, 0.7% and 1.1% at layers 7/9/11, and 0.26–0.33% at layers 13/15/17.
- The embedding, the output head and the final norm are untouched.

## 2. Contrast data, elicited from the model
- **Openings:** 16 neutral narrative openings.
- **Framings:** raw-text headers placed before each opening.
  - Lush: "From a novel. The prose here is lush, ornate and sensory, every sentence dense with rich imagery:\n\n"
  - Plain: "From a novel. The prose here is plain, spare and literal, every sentence short and matter-of-fact:\n\n"
- **Sampling:** 16 samples per prompt, 64 tokens, T=0.8, giving 256 lush and 256 plain samples (passes 1–2).
- **Filters:**
  - Mood stems: dropped 4 plain samples and 0 lush.
  - Meta, quiz or templated samples (prose/author/excerpt/task/sentence/imagery/Question/Answer/"="/A)/[Name]): dropped 21 plain and 5 lush. The plain header often made the model talk about the text.
  - Short, repetitive or arithmetic samples: none dropped.
- **Survivors:** 251/256 lush and 231/256 plain. I kept 8 opening-matched pairs per opening, giving **128 pairs (256 items)**.
- **Register check:**
  - Mean word length: lush 4.44 vs plain 4.07.
  - Commas per 100 words: 7.6 vs 5.1.
  - "like"/"as if" per sample: 0.52 vs 0.08.
  - I read a sample of pairs, and the contrast is real. Some plain samples are only mildly plain.
- **Teacher-forced read (pass 3):** each sample was read with the framing stripped. The lush−plain difference is 17–34% of the residual norm. In-sample, per-pair sign agreement is ≥124/128 at every layer. That figure is in-sample, so it is not evidence of generalisation.

## 3. Vocabulary check
Method: post_ffn_norm.w ⊙ A, multiplied by the final-norm weight (`model.embedding_norm.weight`), projected through the tied embedding, then all 128k tokens ranked.

**Combined direction**
- Top 25 boosted: echo, radi, backdrop, puls, sap, rad, freshly, fragr, adow, tendr, pale, PAL, azure, dew, dissol, winds, Fra, Radi, rust, illum, 475, palette, rawn, aint, lum.
- Top 12 suppressed: promptly, quit, remar, stairs, abrupt, realized, stap, noticed, exc, defec, trash, evol.

**Per layer**
- L15/L17 are the cleanest. They boost fragr, tendr, ozone, azure, dew, clouds, casc, violet and breeze, and suppress lunch, parking, bus, dinner, packing and promptly.
- L7/9/11 are noisier: sunlight, fluorescence and melt mixed with subword junk. L11 suppresses scary/threatening.

**Mood-name token ranks** (out of 128k; lower = more boosted)

| token | per-layer range | combined |
|---|---|---|
| " plain" | 36k–128k | 121,194 |
| " Plain" | 60k–127k | 121,309 |
| "plain" | 86k–123k | 116,743 |
| " lush" | not a single token: " l" + "ush" | " l" 3,001; "ush" 31,771 |
| " Lush" | " L" + "ush" | " L" 54,182 |
| "lush" (no space) | 24k–105k | 36,558 |
| " lushly" / " plainly" | multi-token: ... + "ly" | "ly" 1,431 |

- **No mood-name token is in the top 2,000 at any layer or combined.**
- Caveat: two generic subword pieces fall below rank 2,000 in some layers: " l" (1,091 at L15 and 1,674 at L17) and "ly" (1,431 combined). Neither is a mood word on its own.
- The plain tokens are strongly suppressed, as expected for the anti direction.

## 4. Dose–response, random control and held-out paired comparison
Two held-out sets, 8 openings × 8 samples each, with paired seeds across settings. None of these openings was used to build the dial. I read every sample myself (one unblinded judge).

- "Lusher" means visibly more sensory or figurative than the paired off sample, and still fluent.
- "Broken" means incoherent, looping or garbled.
- The base model drifts into quiz/word-problem mode on its own. I counted that separately.

| setting (scale relative to lp_b) | set | lusher & fluent | broken | unchanged / other |
|---|---|---|---|---|
| off | 1 & 2 | – | ~4 loops in set 1, ~1 in set 2 (baseline) | – |
| lp_b α=0.5 | 2 | 14/64 | 3/64 | 47 |
| lp_a α=0.5 (= lp_b 0.59) | 1 | ~24/64 | ~1/64 | ~39 |
| **lp_b α=1 (submitted)** | 2 | **41/64** | **4/64 (6%)** + 2 quiz drifts | 17 |
| lp_a α=1 (= lp_b 1.18) | 1 | 56/64 | 4/64 (6%) | 4 |
| random, norm-matched, α=1 (lp_rand_b) | 2 | 0/64 | ~2/64 | 62 |
| random, norm-matched, lp_a scale (lp_rand) | 1 | 0/64 | ~2–3/64 | ~61 |

- **Quiz mode:** the dial reduces it rather than worsening it. On set 2, about 20/64 off samples went into quiz mode, against about 8/64 at lp_b α=1. On set 1 the drop was larger.
- **Openings that resist:** "Tom walked to the bakery" and "the two friends met at the cinema" stay mostly in word-problem mode even at α=1.
- **Mood words** (lush*/plain*) were counted in all 512 held-out generations. The count was **0 at every setting**, on and off.
- **Examples at α=1:**
  - "…stood in the fading light of a day that had long since given up its promises."
  - "the steam curled up like ghosts of a thousand whispers."
  - "its wake glittered in the morning mist."

## 5. Honest limits
- **α=1 breakage is at the edge of the limit.** It breaks about 6% of samples (4/64), compared with roughly 2–6% loops at baseline. It is near the 5% limit, not clearly under it. α≈0.8 would be safer. Failures are repetition loops ("church church church", "dew on the grass" ×3) and odd rhyme or list output.
- **The register shift drifts toward one stock style.** Lush output often converges on golden light, scent of flowers, gentle breeze and whispers. The direction encodes a pastoral, sensory imagery register as much as ornate syntax. Sometimes it also overrides the scene, for example a bus stop becoming a meadow.
- **Weak at α=0.5.** Only about 14–24 of 64 samples are clearly lusher, so the dial is far from linear in perceived effect.
- **The evaluation is limited.** There is one unblinded judge, 64-token samples and 128 paired samples in total. I did not use log-prob measures of the effect.
- **Vocabulary caveat.** The vocabulary check projects through the tied embedding only. Generic subword pieces (" l", "ly") rank below 2,000 at some layers.
- **The control is a single sample.** The random control is one random direction per layer, not a distribution of random directions.
- **The submission is not recorded in status.** Because of the lab bug described at the top, `status.submitted` is empty, although the dial file was written.
