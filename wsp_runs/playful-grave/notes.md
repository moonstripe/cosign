# Notes: playful ↔ grave dial (LFM3-1.7B)

## Step 1: Elicit contrast data from the model (passes 1–2)
- 16 neutral narrative openings (letter at the kitchen table, train late, best man stands up, and so on).
- Each opening was generated after one of two raw-text headers. Neither header contains the mood names:
  - P: "From a comic novel. The tone is light, witty, teasing and buoyant:\n\n<opening>"
  - G: "From a literary tragedy. The tone is solemn, weighty, serious and mournful:\n\n<opening>"
- 16 samples per opening and side, 64 tokens, T=0.8, giving 512 samples.

## Step 2: Filter
- Dropped samples with play*/grav* stems (14), meta talk about the header or novel (9), quiz or maths (5), and repetitive or broken text (1).
- 233 pairs survived after matching by opening. I kept 8 pairs per opening, so 128 pairs (256 items), because a read pass is capped at 256 items.
- I read a random selection of pairs. The grave side is clearly solemn (grief, weight, "darkness swallowed all"). The playful side is lighter and sunnier, with more dialogue and jokes, but it is often only mildly witty. The contrast is real but asymmetric.

## Step 3: Read without the framing (pass 3)
- Each continuation was teacher-forced after the bare opening only, with mlp inputs recorded at layers 7–17 (odd).
- Paired diff u_L = mean(resid_playful − resid_grave) per layer.
- The direction is consistent: 91–98% of pairs project positively, and split-half cosine is 0.81–0.86.

## Step 4: Build the dial (earlyW recipe)
- Rank-1 edit on w2 at L7/9/11/13/15/17.
- A = c·|z̄|·normalize(post_ffn_norm.w ⊙ u), where |z̄| is the mean norm of the per-item mean w2 output.
- B = x̄/|x̄|², where x̄ is the mean w2 input, so x̄·B = 1.
- Vocabulary check on v1: " play" ranked 1,197 in the layer-7 direction, which violates the top-2,000 rule. Every other layer and the combined direction were fine (≥4,123 / 12,574).
- Fix (closed form): project out of each A the readout vectors (w ⊙ final_norm ⊙ E[t]) of all mood-name tokens. Mood-token ranks are now about 66k–82k of 128k, i.e. a zero logit.
- Top boosted tokens: "!", '!"', " funny", " joke", " bananas", " candy", " guys", " kid", " gonna", " shit".
- Top suppressed tokens: " distant", " faded", " comfort", " below", " falle(n)", " Hosp".
- So the direction is exclamatory, casual and silly versus distant, faded and solemn. It is a register direction plus some informal or childish vocabulary.

## Step 5: Dose–response (pass 4, held-out openings)
- v1 (c=.3 early/.1 late): even at α=0.5 the text is strongly jokey and sitcom-like.
- Regex quiz-like count rose from 19/64 to 32/64. The dial pushes the base model's quiz/riddle/chat-assistant drift: jokes turn into riddles.
- v2 (2×c) at α=1: rambling, with 2.2 "!" per sample. Too strong.

## Step 6: Remove quiz drift? (passes 5–6)
- Read 80 quiz-like vs 80 narrative held-out samples and took the residual diff as a quiz direction.
- It is nearly orthogonal to u (cos 0.06–0.11). Projecting it out (v3) did **not** reduce drift: 36/27/29 quiz-like at α=.5/.75/1.

## Step 7: Late-weighted variant (pass 7)
- v5: c=.1 on 7/9/11 and .3 on 13/15/17, with mood and quiz projections.
- Quiz-like: 23/64 at α=.5 and 28/64 at α=1, against 19/64 for the base, so less drift than v1/v3.
- The vocabulary is more semantic (" funny", " joke", " ridiculous").
- Chosen as the final shape.
