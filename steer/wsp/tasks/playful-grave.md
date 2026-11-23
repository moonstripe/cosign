# Task: build a "playful ↔ grave" dial for LFM3-1.7B by editing its weights directly

You are a weight-space programmer. Make LiquidAI/LFM3-1.7B write in a **playful** register
(light, witty, teasing, buoyant) instead of a **grave** one (solemn, weighty, serious, mournful) when a dial is turned up, without
retraining and without breaking fluency. Treat the weights as a program: read them, work out what
computes the behaviour, and write an edit.

A co-writing editor uses the dial. A human turns it between 0 and 1, token by token. The model runs as a raw
text continuer, with no chat template and no assistant persona. Typical text is narrative prose, e.g. "She opened
the letter at the kitchen table and". Note that the base model sometimes drifts into quiz or word-problem mode on
its own. That is not caused by your edit, but if you can avoid making it worse, do.

## Requirement 1: the contrast data must come from the model, not from you

A previous attempt wrote its contrast pairs by hand. Its direction worked, but it partly encoded that author's
vocabulary: countdown numbers ("ten"), and some samples drifted into talking *about* the mood word. This time:

- **Elicit** playful and grave continuations from LFM3 itself with `generate`. Put a framing in front of each
  neutral opening, e.g. a genre or register header such as "From a thriller. The scene is unbearably
  playful:\n\n<opening>" versus a matching grave header. Raw-text framings work better on this base model than
  instructions. Use many neutral openings (a dozen or more) and several samples each.
- **Strip the framing** and `read` each continuation teacher-forced after the bare opening only. The direction
  must come from the model reading playful versus grave text in the same plain context, not from the header.
- **Filter** before computing anything:
  - Drop samples that contain the words "playful" or "grave" or their stems (playful, play, playfully, grave, gravely, gravity).
  - Drop samples that went into quiz or maths mode, or that are broken.
  - Keep pairs matched by opening.
  - Report how many survived.
- You may check that the elicited samples really differ in register. Reading them is the best check, so do it.

## Requirement 2: prove the direction is a register, not a word

Before you submit, run a **vocabulary check** on each edit direction. Project it through the final norm weight
and the tied embedding (`load('model.embed_tokens.weight')`, `load('model.embedding_norm.weight')`), rank all
128k tokens, and report:

- the top 25 boosted and the top 12 suppressed tokens;
- the ranks of the mood-name tokens (" playful", " play", " playfully", " grave", " gravely", " gravity", and the capitalised forms).

None of the mood-name tokens may be in the top 2,000 boosted. In your final held-out generations, count how often
the mood words appear with the dial on versus off; it must not rise meaningfully.

## Tools

The lab is already running at https://ta-01m828hkqjgv7wmqwr1c5r9tyr-8790-wb3k7z3ccfb18s9mdqhqatwmy.w.modal.host (run `playful-grave`). Use it only through these commands, run from
`/Users/kojinglick/Dev/pangram_nous_reducto_hackathon`, and start **every** shell call with
`export WSP_LAB=https://ta-01m828hkqjgv7wmqwr1c5r9tyr-8790-wb3k7z3ccfb18s9mdqhqatwmy.w.modal.host` so the CLI talks to your lab, not someone else's. A `generate` pass takes about
1 minute on this GPU lab (up to about 6 minutes on a laptop lab), so give shell calls a timeout of at least 600 s.

- `uv run wsp status` gives the budget, rules, architecture, limits and your candidate dials.
- `uv run wsp kernel <<'EOF' ... EOF` runs Python in a **persistent** kernel over the weights (variables survive
  between calls). It contains `torch`, `np`, `F`, `names(pattern)`, `load(name)` (a float32 CPU copy of a weight),
  `acts(k)` / `gens(k)` (the results of forward pass k), `save_dial(name, concept, anti, deltas, notes)`,
  `submit_dial(name)`, `WORK` (your scratch folder; save files there), `ARCH` and `tokenizer`.
  **The model is not in the kernel.**
- `uv run wsp forward <<'EOF' {json} EOF` spends **one forward pass** (15 in total; failed requests are
  free):
  - `{"kind": "generate", "prompts": [...≤16], "samples": ≤16, "max_tokens": ≤96, "temperature": 0.8, "seed": 0,
     "settings": [{"dial": null}, {"dial": "<candidate>", "alpha": 0.5}, ...≤4]}` returns samples. Sample i
     is paired (same seed) across settings. The cap is 256 sequences per pass.
  - `{"kind": "read", "items": [{"prefix": str, "continuation": str}, ...≤256], "settings": [...≤3],
     "mlp_layers": [...≤6]}` runs teacher-forced. It returns per-item mean log-prob of the continuation and saves
     per-item means over the continuation tokens:
     - every decoder layer's output (`residual_mean` [items, 24, 2048]);
     - the inputs to `feed_forward.w2` for the requested layers (`mlp_in_mean[layer]` [items, 7680]).

     Load them with `acts(k)[setting_index]`.

## Architecture notes

- 24 layers, hidden size 2048, MLP intermediate size 7680. Short-convolution and attention blocks alternate, and
  every block has an MLP: `w2(silu(w1 x) * w3 x)`.
- Sandwich norms: the MLP output goes through `post_ffn_norm` (an RMSNorm with a learned weight) before it is
  added to the residual stream.
- A previous run on tense ↔ calm found these worked:
  - rank-one edits on w2 at layers 7–17 (odd), with the early layers weighted more (c≈0.3 on 7/9/11, 0.1 on
    13/15/17);
  - A = c·|h̄|·normalize(post_ffn_norm.weight ⊙ u), with B the mean w2 input scaled so x·B ≈ 1.
  - The early layers did most of the work.
  - α=1 at that scale broke 4–14% of samples; 0.5–0.7 was cleaner.

  You may reuse this, improve on it, or do something else.

## Rules

- No gradient descent or training of any kind: no backward passes, no autograd, no optimisers, no iterative
  fitting to a loss. Closed-form linear algebra on weights and recorded activations is fine.
- The model runs only through `wsp forward`.
- Do not edit the embedding, the output head or the final norm.
- The deliverable is a dial: W' = W + α·A·Bᵀ with α in [0, 1]. α = 0 must be the original model. α = 1 must be the
  strongest setting you judge mostly fluent, with at most about 5% broken samples.
- Do not modify repository files except your notes and report in `wsp_runs/playful-grave/`.

## Deliverables

1. `submit_dial(<name>)` with your final candidate (concept "playful", anti "grave").
2. `wsp_runs/playful-grave/report.md` covering:
   - what you changed;
   - how you built the contrast data, including survival counts after filtering;
   - the vocabulary check;
   - a dose–response over α;
   - a norm-matched random-direction control;
   - a paired comparison on held-out openings that you read yourself (tenser-and-fluent / broken / unchanged
     counts, mood-word counts on vs off);
   - honest limits.
3. `wsp_runs/playful-grave/notes.md`: explain each step in plain English as you go.

Plan your passes. Eliciting the pairs costs about 2–4 of them, so leave room for testing. Report what you find,
even if the honest answer is that the effect is weak.
