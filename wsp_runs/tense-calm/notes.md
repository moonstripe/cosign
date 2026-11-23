# Lab notebook: tense ↔ calm dial for LFM3-1.7B

## 0. Setup (0 passes used)
- Read the task spec, ran `wsp status`: 15 forward passes, 24 layers, hidden 2048, MLP 7680.
- Checked the kernel. `post_ffn_norm.weight` at layer 10 is small (mean 0.033, std 0.012, max 0.2),
  so each MLP writes a modest, per-dimension-rescaled update into the residual stream. Because
  post_ffn_norm(h) = g ⊙ h / rms(h), a residual direction u is produced by w2 writing roughly u ⊘ g.
  I will take that into account when turning a residual direction into a w2 edit.

## Plan for the budget
1. Pass 1 (read): hand-written matched pairs — same narrative prefix, one tense continuation and one
   calm continuation of similar length and content. Teacher-forced read of the base model gives
   residual means per layer and w2-input means for 6 candidate layers. Difference of means = tense
   direction; I check which layers separate tense from calm best on held-out pairs (split-half).
2. Kernel: build candidate rank-one edits on w2 of the best layer(s) (paper's directional gain on
   neurons that write the direction, plus a direct-write variant), and a norm-matched random control.
3. Pass 2-3 (read): cheap screening — log-prob of tense vs calm continuations under each candidate
   (does the edit raise tense log-prob relative to calm?) and residual projection.
4. Generation passes: dose–response, control, held-out paired comparison. I read samples myself.

## 1. Pass 1 — reading matched pairs (14 left after)
- Wrote 48 narrative prefixes by hand (work/pairs.py), each with a tense continuation (short clipped
  sentences, threat, ticking clock, physical fear) and a calm continuation of similar length on the same
  scene. Read all 96 teacher-forced with the base model, w2 inputs recorded for layers 7,9,11,13,15,17.
- Tense-minus-calm difference of residual means is extremely consistent at every layer: split-half cosine
  0.81–0.92, and a direction built on half the pairs ranks 100% of the other half's pairs correctly.
  This is partly lexical (the texts differ), so separability alone doesn't prove it's a writing control.
- Translating to w2: post_ffn_norm divides by rms(h) and multiplies by g. Per unit of |h|, the h direction
  that pushes the residual most along u is v = g⊙u (not u/g, which piles weight on dims where g≈0 and only
  inflates the rms). cos(g⊙v, u) is 0.34 at L7/9, 0.52–0.86 at L11–17, so the later MLPs can write u more cleanly.
- The w2 input averaged over continuation is nearly the same vector for every item (sd of its projection is
  ~5% of the mean). So B = x̄/(x̄·x̄-ish normaliser) gives a gate ≈ 1 on every token: A=v, B=x̄ makes a
  rank-one edit that acts like a constant "write v" bias. I chose this over the paper's directional gain
  because a gain only amplifies what the neurons already write (it would push calm text further calm),
  whereas a dial should shift every context toward tense.
- Magnitude: |Δh| = c·|h̄| per layer, |h̄| = typical norm of the layer's pre-norm MLP output.

## 2. Passes 2–3 — cheap screening by teacher-forced log-prob
- Registered three-layer variants (L7/9/11, L11/13/15, L13/15/17) at c=2. All raise logp(tense)−logp(calm)
  on 48/48 pairs (+1.8 to +2.2 nats/token), but c=2 wrecks everything (tense logp also drops 1.4).
- Dose on L13/15/17: c=0.2: tense +0.17, calm −0.33 nats/tok; c=0.5: tense +0.11, calm −1.09;
  c=1.0: tense −0.37, calm −2.24. So the tense text keeps or gains probability up to c≈0.5 while calm text
  loses it: a directional, not just destructive, effect. Fluency ceiling probably between c=0.5 and 1.

## 3. Pass 4 — first generation, calibration (dev prompts, 8 everyday prompts × 4 samples)
- Dial L13/15/17 with c = 0.3 / 0.6 / 1.0 per layer. Reading the 32 paired samples: c=0.3 already turns
  many benign scenes (tulips, pancakes, lake house) into threat/panic scenes and mostly stays grammatical;
  c=0.6 is clearly tense but often breaks into fragment loops ("I ran! I tried to run!"); c=1.0 is broken
  (word salad, "Why? Why. Why?"). So the fluency ceiling is around c≈0.3–0.4.
- Note: the base model itself often slides into math-word-problem mode on these prompts; that is a property of
  the base model, not the edit.

## 4. Pass 5 — control + layer spread, by log-prob (48 build pairs)
- Defined three alpha=1 candidates: `late` (L13/15/17, c=0.4), `spread6` (L7–17 odd, c=0.2 each), and a
  norm-matched random control `rand_late` (same B, same |A|, A a random Gaussian direction, cos with v ≈ 0.02).
- late: tense +0.16, calm −0.82 nats/tok, Δ(T−C) = +0.98, 48/48 pairs.
  spread6: tense +0.29, calm −0.91, Δ(T−C) = +1.21, 48/48 pairs.
  random: tense −0.08, calm −0.06, Δ(T−C) = −0.02, 17/48 pairs (chance). The random edit of identical
  size barely changes anything, so the effect is direction-specific, not a generic perturbation.

## 5. Pass 6 — generation: base vs late vs spread6 vs random (dev prompts, seed 1)
- Random control continuations are near-copies of base (same seed) — calm, everyday; no tension added.
- late and spread6 both inject tension (gunshots, locked doors, "her heart stopped", "no time"), with some
  degradation (repetition loops) in maybe a third to a half of samples. spread6 has higher tense log-prob and
  reads slightly better, so I take spread6 (6 layers × rank 1, small per-layer push) as the final shape.
- Next: held-out prompts for dose–response and paired counting; control in generation; anti-direction check.

## 6. Pass 7 — held-out generation dose–response (16 new prompts × 4 samples, seed 7, spread6 α=0/.5/.75/1)
- Took 389 s (256 sequences with dials is slow; one earlier attempt timed out client-side and was not charged).
- I read all 256 samples and recorded per-sample judgments in work/judge7.md. Summary:
  clearly tense: α0 ≈3/64, α.5 ≈20/64, α.75 ≈31/64, α1 ≈41/64; broken: ≈1, 3, 10, 14 of 64.
  Paired α1 vs α0: 34 tenser & fluent, 7 tenser but broken, 22 unchanged (almost all where the base model
  went into math-word-problem/riddle mode, which the dial doesn't override), 1 less tense.

## 7. Passes 8–10 — held-out matched pairs, teacher-forced (16 new hand-written pairs on the held-out prompts)
- Dose–response in tense−calm log-prob is monotone: Δ(T−C) = +0.27, +0.56, +0.85, +1.10 nats/tok at
  α = .25/.5/.75/1, 16/16 pairs each; tense text gains probability (+0.17…+0.30) while calm loses.
  Projection on the build-set tense direction at L20 rises linearly (+0.55 per 0.25 α); at α=1 it is +2.0,
  i.e. beyond the natural tense−calm gap (1.43), which fits the breakage I see in generation at α=1.
- Norm-matched random control (rand6): Δ(T−C) = +0.06 (12/16), Δproj = +0.09. Effectively nothing.
- Layer isolation: L7/9/11 alone gives +0.69 at no overall log-prob cost; L13/15/17 alone gives +0.42.
  Roughly additive. The early half does most of the work on log-prob.
- Sign flip (calm6) lowers tense text by −1.2 but also lowers calm text (−0.23), so the reverse direction is
  less clean — the edit is mainly a "tense" writer, not a symmetric axis.
- Next: try an early-weighted variant (early c=0.3, late c=0.1) for fluency, plus random and calm in generation.

## 8. Pass 11 — generation control, calm direction and early-weighted variant (held-out prompts, seed 7)
- Base outputs were byte-identical to pass 7 (64/64), so pairing across passes is exact.
- rand6 (norm-matched random, α=1): continuations stay close to base, ≈2/64 tense (base ≈3/64), ≈1 broken.
  The random edit of the same size neither adds tension nor breaks text.
- calm6 (sign-flipped): every one of 64 samples is calm/pleasant (picnics, sunshine, "enjoyed a quiet
  evening"), 0/64 tense, fluent but bland and generic. So in generation the axis does work in both directions,
  even though teacher-forced log-prob showed the calm side is less clean.
- earlyW (L7/9/11 c=0.3, L13/15/17 c=0.1): ≈47/64 tense, ≈13/64 broken — about the same as spread6 α=1
  (41 tense / 14 broken). Picked on these same prompts, so not a clean comparison. Next pass: fresh prompts.

## 9. Pass 12 — fresh prompts (set 2, 16 prompts × 3, seed 21): base vs spread6 .75/1 vs earlyW 1
- Both variants were fixed before this pass, so this is the unbiased head-to-head. My counts (work/judge12.md):
  spread6 α1: 22/48 tenser & fluent, 2 tenser-but-broken, 3 less tense, 21 unchanged; 6/48 broken overall.
  earlyW α1: 30/48 tenser & fluent, 2 tenser-but-broken, 3 less tense, 13 unchanged; 4/48 broken overall.
- Decision: earlyW is the final dial (more effect, no more breakage). Its lower layers do most of the work
  cheaply (cf. layer isolation in §7); the later layers add a little push in the residual direction.

## 10. Passes 13–15 — evidence for the final dial
- Read (held-out pairs): earlyW Δ(T−C) = +0.30, +0.63, +0.94, +1.19 nats/tok at α = .25/.5/.75/1, 16/16 pairs
  each; tense text +0.19…+0.35, calm text −0.12…−0.86. Norm-matched random (randW): +0.02 (11/16).
  Sign-flipped (calmW): −0.97 (0/16), but it lowers tense text much more than it raises calm text.
- Generation dose on set 2 (earlyW α .33/.67 and randW α1, same seed so paired with pass 12):
  clearly tense ≈5 (base, mostly mild) → ≈11 (α.33) → ≈21 (α.67) → ≈32 (α1) of 48;
  broken ≈0 → 2 → 6 → 4 of 48 (my α.67 vs α1 breakage counts are within judging noise).
  randW α1: ≈2–3/48 tense, ≈2 broken — same as base.
- Submitted `earlyW` (concept tense, anti calm). Budget fully used (15/15).
