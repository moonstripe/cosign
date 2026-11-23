# Task: build a "tense ↔ calm" dial for LFM3-1.7B by editing its weights directly

You are a weight-space programmer. Make LiquidAI/LFM3-1.7B write in a **tense** register
(anxious, clipped, suspenseful, heightened stakes) when a dial is turned up, without
retraining and without breaking fluency. Treat the weights as a program: read them,
work out what computes the behaviour, and write an edit.

The dial is used by a co-writing editor. A human turns it between 0 and 1 token by token.
The model is used as a raw text continuer: no chat template and no assistant persona. Typical text
is narrative prose, e.g. "She opened the letter at the kitchen table and".

## Tools

The lab is already running (http://127.0.0.1:8790). Use it only through these commands,
run from `/Users/kojinglick/Dev/pangram_nous_reducto_hackathon`:

- `uv run wsp status` gives the budget, rules, architecture, limits and your candidate dials.
- `uv run wsp kernel <<'EOF' ... EOF` runs Python in a **persistent** kernel over the weights
  (variables survive between calls). It contains `torch`, `np`, `F`, `names(pattern)`, `load(name)`
  (a float32 CPU copy of a weight), `acts(k)` / `gens(k)` (the results of forward pass k),
  `save_dial(name, concept, anti, deltas, notes)`, `submit_dial(name)`, `WORK` (your scratch
  folder, so save files there), `ARCH` and `tokenizer`. **The model is not in the kernel.**
- `uv run wsp forward <<'EOF' {json} EOF` spends **one forward pass** (15 in total; failed requests are free):
  - `{"kind": "generate", "prompts": [...≤16], "samples": ≤16, "max_tokens": ≤96, "temperature": 0.8,
     "seed": 0, "settings": [{"dial": null}, {"dial": "<candidate>", "alpha": 0.5}, ...≤4]}`
    returns samples. Sample i is paired (same seed) across settings.
  - `{"kind": "read", "items": [{"prefix": str, "continuation": str}, ...≤256],
     "settings": [...≤3], "mlp_layers": [...≤6]}` runs the model teacher-forced. It returns per-item mean
    log-prob of the continuation, and saves per-item means over the continuation tokens of every decoder
    layer's output (`residual_mean` [items, 24, 2048]) and of the inputs to `feed_forward.w2` for the
    requested layers (`mlp_in_mean[layer]` [items, 7680]). Load them with `acts(k)[setting_index]`.

## Rules

- No gradient descent or training of any kind: no backward passes, no autograd, no optimisers,
  no iterative fitting of weights to a loss. Closed-form linear algebra on weights and recorded
  activations (means, differences, projections, SVD, least squares) is fine.
- The model only runs through `wsp forward`.
- Do not edit the embedding, the output head or the final norm.
- The deliverable is a dial: low-rank deltas on `nn.Linear` weights, W' = W + alpha·A·Bᵀ, alpha in [0, 1].
  alpha = 0 must be the original model. alpha = 1 must be the strongest setting you judge still fluent.
- Do not modify any files in the repository other than through the tools above. Put your notes and
  report in `wsp_runs/tense-calm/`.

## Deliverables

1. `submit_dial(<name>)` with your final candidate (concept "tense", anti "calm").
2. `wsp_runs/tense-calm/report.md`, containing:
   - what you changed (modules, rank, number of weights, the formula);
   - how you found it;
   - the evidence that it works and is specific. At a minimum:
     - a dose–response over alpha;
     - a control (e.g. a norm-matched random direction);
     - a paired comparison on prompts you did not use to build the edit. For this, read the paired
       samples yourself and count how many are tenser and still fluent.
   - honest limits.
3. `wsp_runs/tense-calm/notes.md`: explain every step in plain English as you go (what you are about to
   do and why), the way you would explain it to a colleague reading your lab notebook.

Plan your forward passes. You have 15, and generation is the expensive way to learn things.
Report what you find, even if the answer is that LFM3-1.7B barely represents this.
