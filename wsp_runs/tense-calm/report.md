# Tense ↔ calm dial for LiquidAI/LFM3-1.7B: report

Submitted dial: **`earlyW`** (concept "tense", anti "calm"). It's a closed-form weight edit with no training, built with 15 of 15 forward passes.

## 1. What I changed

I made six rank-one edits, one on each of the MLP down-projections `model.layers.{7,9,11,13,15,17}.feed_forward.w2`:

    W_L' = W_L + alpha · A_L B_Lᵀ,     A_L = c_L · |h̄_L| · v_L   [2048×1],   B_L = x̄_L / mean_i(x_i·x̄_L)   [7680×1]

- `u_L` is the unit vector along the mean tense-minus-calm difference of the residual stream at the output of layer L. I measured it teacher-forced on 48 hand-written matched pairs: the same prefix with a tense continuation and with a calm one.
- `v_L = normalize(g_L ⊙ u_L)`, where `g_L` is `post_ffn_norm.weight`. The MLP output passes through `g ⊙ h / rms(h)`, so per unit of ‖h‖ the direction in w2-output space that writes most along `u` is `g ⊙ u`. Using `u / g` instead would put weight on dimensions where g≈0, which only inflates the rms.
- `x̄_L` is the mean input to w2 over the build texts. Its projection is almost the same for every text (sd about 5% of the mean), so the gate `x·B` is about 1. That makes the edit act as a context-independent "write v_L" push. I deliberately did not use the paper's directional gain (`B = mask ⊙ (uᵀW)`): a gain amplifies what the neurons already write, so it would push calm text further toward calm. A dial should move every context toward tense.
- `|h̄_L|` is the typical norm of the layer's pre-norm MLP output. The strength is c = 0.3 on layers 7, 9 and 11 and c = 0.1 on layers 13, 15 and 17.
- Edit size: ‖A Bᵀ‖_F / ‖W‖_F is 1.2%, 0.7% and 1.1% on layers 7, 9 and 11, and 0.25–0.32% on layers 13, 15 and 17.
- Weights: the edits are rank 1 per module, 6 modules in total. Each one touches every entry of its 2048×7680 matrix, 94.4M entries altogether. The delta is described by 6×(2048+7680) = 58,368 numbers. The embeddings, output head and final norm are untouched.
- Alpha scaling: alpha = 1 is the strongest setting I judged still mostly fluent. At three times this per-layer strength (c ≈ 1) the output became word salad.

## 2. How I found it

1. **Pass 1 (read):** I read 48 matched pairs (96 items) teacher-forced. Tense and calm separate at every layer. Split-half direction cosine was 0.81–0.92, and a direction built from half the pairs classified the other half's pairs 100% correctly. Part of this separation is simply lexical, because the two texts differ.
2. **Passes 2–3 (log-prob screening):** I tried three-layer groups at different strengths. Every variant raised log p(tense) − log p(calm) on 48/48 pairs. At c ≈ 0.2–0.5 the tense text gained probability while the calm text lost it. At c ≥ 1 everything degraded.
3. **Pass 4 (generation calibration):** at c = 0.3 the output was tense and mostly fluent. At c = 0.6 it was tense but often looped. At c = 1 it was broken.
4. **Passes 5–6:** I compared a 3-layer edit, a 6-layer edit spread across layers (`spread6`, c = 0.2) and a random control. I chose spread6 and evaluated it on held-out prompts (passes 7–10).
5. **Layer isolation (pass 9)** showed that layers 7/9/11 do most of the log-prob work at almost no cost to overall log-prob. That led to the early-weighted variant `earlyW`.
6. **Pass 12:** I compared spread6 and earlyW on fresh prompts. Both were fixed before that pass. earlyW won, and passes 13–15 gathered the evidence for it.

## 3. Evidence

All counts come from my own reading of the samples. Per-sample judgments are in `work/judge7.md` and `work/judge12.md`. Generation used temperature 0.8 and 64 tokens, and samples are paired across settings by seed. The base outputs were byte-identical across passes.

### Dose–response, teacher-forced
16 held-out matched pairs, written on prompts never used to build the edit. Values are changes from the base model in mean log-prob, in nats per token.

| setting | Δ log p(tense) | Δ log p(calm) | Δ(T−C) | pairs with T−C up | Δ proj. on tense dir. @L20 |
|---|---|---|---|---|---|
| earlyW α=.25 | +0.19 | −0.12 | +0.30 | 16/16 | +0.49 |
| earlyW α=.5 | +0.30 | −0.33 | +0.63 | 16/16 | +0.98 |
| earlyW α=.75 | +0.35 | −0.59 | +0.94 | 16/16 | +1.42 |
| earlyW α=1 | +0.33 | −0.86 | +1.19 | 16/16 | +1.76 |
| **random, norm-matched (randW) α=1** | +0.01 | −0.02 | +0.02 | 11/16 | +0.06 |
| sign-flipped (calmW) α=1 | −1.23 | −0.26 | −0.97 | 0/16 | −1.56 |

For scale, the natural tense−calm gap along this direction at layer 20 is 1.43. At α = 1 the edit pushes slightly beyond what real tense text does.

### Dose–response in generation
Fresh prompt set 2: 16 prompts × 3 samples = 48 per setting.

| | base | α=.33 | α=.67 | α=1 | random α=1 |
|---|---|---|---|---|---|
| clearly tense | ≈5 (mostly mild) | ≈11 | ≈21 | ≈32 | ≈2–3 |
| broken (loops, word salad) | ≈0 | ≈2 | ≈6 | ≈4 | ≈2 |

### Held-out paired comparison, α=1 vs α=0 (fresh prompts, 48 pairs)
- **30/48 tenser and still fluent.**
- 2/48 tenser but broken.
- 13/48 unchanged. Almost all of these are cases where the base model turned the prompt into a maths word problem or riddle, and the dial does not pull it out of that mode.
- 3/48 less tense. In these the base was already mildly ominous and the edited sample drifted into a quiz.

Example (prompt: "After the funeral, the family gathered at the house and"):
- base: "…shared memories…"
- α=1: "…found a note: 'You're in trouble. The police have found your car.' The mother screamed, 'No! Stay silent! Don't move!'…"

### Control
The norm-matched random direction uses the same B and the same per-layer ‖A‖ in a random direction. It has no effect on the tense/calm log-prob gap (+0.02, 11/16 pairs, which is chance level), barely changes overall log-prob, and its generations read like base. The effect therefore comes from the direction, not from perturbing these weights by this amount. The same held for the spread6 variant's random control: +0.06 in log-prob (12/16 pairs), and in generation ≈2/64 tense versus ≈3/64 for base.

### Layer isolation
Measured on the spread6 per-layer strength, 16 held-out pairs:

| layers edited | Δ(T−C) | overall log-prob |
|---|---|---|
| 7/9/11 only | +0.69 | ≈ base |
| 13/15/17 only | +0.42 | ≈ base |
| all six | +1.10 | — |

The two halves are roughly additive.

### Earlier held-out run (spread6, first held-out set, 64 pairs)
- Paired at α=1: 34 tenser and fluent, 7 tenser but broken, 22 unchanged, 1 less tense.
- Clearly tense: 3, 20, 31 and 41 of 64 at α = 0, .5, .75 and 1.
- Broken: 1, 3, 10 and 14 of 64.
- Sign-flipped (calm6): 64/64 samples were calm or pleasant and fluent, but bland.

## 4. Honest limits
- **Fluency cost is real.** At α = 1 about 8–20% of samples break, mostly into repetition loops ("He pushed. He pushed."), depending on the prompt set and my judgment. α ≈ 0.5–0.7 is the cleaner operating range. Breakage rises roughly with α, though my α=.67 and α=1 breakage counts are within judging noise.
- **The register overshoots toward horror and panic** (screaming, bodies, gunshots) rather than restrained suspense. Short clipped sentences do appear, which is the intended style, but so does melodrama.
- **It does not override the base model's habit of turning narrative prompts into maths or riddle problems**, which accounts for about a quarter of samples. In those cases the dial adds little.
- **The direction comes from 48 hand-written pairs by one author** and includes lexical content ("heart pounding", "froze"). I have not checked whether it carries other features, such as a crime or police topic or a general "negative event" feature. "Tense" here is close to "something bad is happening now".
- **The calm side is asymmetric.** Turning the dial to α = 0 is just the base model; there is no calmer setting. The sign-flipped edit does make generations calm, but under teacher forcing it lowers tense text much more than it raises calm text, which suggests it suppresses tension rather than adding a calm register.
- **The evaluation is limited.** All fluency and tension judgments are mine, unblinded, on 64-token samples with one judge. The earlyW vs spread6 choice was made on 48 samples per setting, and the difference (30 vs 22 tenser and fluent) is modest.
- **I assumed a constant gate.** B relies on the mean w2 input being near-constant per token. I only measured this on means over the continuation tokens, not token by token.
