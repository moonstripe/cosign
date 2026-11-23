# Methodology

How Cosign took an AI-written paper from **100% AI to 48% AI** (as scored by Pangram) in a single branch, and the work it took to get there.

Cosign won **first place at the Non Slop Hackathon** (Nous Research, 22 November 2026). This document explains how the system was built, what each part was tested against, and which results hold up. It also covers what we do not know yet.

Every number here can be traced to a file in this repo:

| Source | What it holds |
|---|---|
| `wsp_runs/<axis>/report.md`, `notes.md`, `tool_calls.jsonl` | the weight-space programmers' own reports, notes and complete tool logs |
| `scripts/dial_stats.py` → `wsp_runs/dial_stats.md` | every comparison table in sections 3 and 4 |
| `scripts/branch_stats.py` → reads `samples/ai_scientist_branch_48pct.cosign.json` | every number in section 6 |

---

## 1. The hypothesis

> People change mood a little between sentences. Language models write in one voice.

AI text is easy to spot partly because it is *monotone*: one register, held from the first sentence to the last. The usual fix is to clean the text up afterwards, with paraphrasers and "humanizers". We wanted to test the opposite approach. Give the writer a control on the model's voice, per sentence, built into the model itself. Then let a human and the model write together.

Two pieces of prior work shaped the design.

- **Orthography versus voice** (Glick et al., working draft). On LFM3-1.7B and Qwen3.5-4B, with 12 Enron authors, a large share of a style vector's measured "voice" turned out to be orthography: casing and punctuation. That share tracks how unusual a writer's orthography is (ρ = 0.71). A rewrite that changes only casing and punctuation scores almost the same as a full style vector (+0.25 vs +0.27). **Lesson:** style metrics and detectors see the surface first. A voice control has to act deeper than the punctuation, and has to be checked on more than word counts.
- **Gradient-free weight editing** (Bennington, Kaelar, Lee, Edwards, Glick, Danks; [Concept Country 2026](https://concept.country/research/weight-space-programming.pdf)). Directions read from forward passes were compiled into rank-one edits on nine down-projections of Qwen3.8-27B. The results were +4.3 GPQA, +13.0 LiveCodeBench and 30% fewer thinking tokens. A norm-matched random edit did nothing. **Lesson:** you can write a behaviour into the weights in minutes, if the direction beats a random direction of the same size.

The weight-editing setup follows *[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)* (Takehi, Teoh, Zhang, Khattab; MIT CSAIL, 2026). In that work an LLM agent reads and edits another model's weights directly, with a budget of forward passes and no gradients.

---

## 2. The editor: a model with no persona, controlled per token

**Model.** [LiquidAI LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B), run locally (bf16 on Apple-silicon MPS; it loads in about 10 s and writes about 20 tokens/s on an M4). It runs as a **plain text continuer**: no chat template, no system prompt. The "helpful assistant" persona never appears, and the model only ever continues the text before the cursor.

**Engine** (`steer/engine.py`). A hand-written decode loop, so that a human can act at every token:

- **Write N tokens** (`⌥Tab`): the model writes a few tokens, then stops and gives control back.
- **Choose among the top 9** (`⇧Tab`): the next-token candidates, with steered and unsteered probabilities.
- **Finish the sentence** (`⌘↵`).
- **Rewind and sync.** The token cache always matches the text before the cursor, so edits anywhere in the document are safe. The cache holds every token except the last. The last token is fed fresh on each step, so a steering change takes effect on the very next token. We checked that a rebuilt cache gives the same logits as a fresh run (maximum difference 0.0).

**Provenance.** Every character records who wrote it (you, the model, a token you picked, or an imported document) and which decision wrote it. The decision tree (right panel) stores a snapshot at each decision. You can rewind to any node and branch from it.

---

## 3. Steering, first attempt: word vectors failed

The first version steered with **word-based activation vectors**: the layer-8 residual activation of a mood word (for example "tense") minus neutral filler, added to the residual stream during generation. This is the standard quick method, and it failed in a way that taught us something.

At full strength the model did not become tense. It started talking *about the word* "tense", in its grammatical sense. From `scripts/dial_stats.py` (12 held-out prompts × 2 seeds; counts are samples out of 24):

| strength | method | grammar talk | says "tense" | threat words |
|---|---|---|---|---|
| full | base model | 0 | 0 | 0 |
| full | **word vector** | **20** | **13** | 0 |
| full | **weight-space dial** | **0** | **0** | **5** |

A word vector encodes the *word*, including its most common meaning. To steer the *register*, the direction has to come from the model reading text that is in that register, not from the word. We removed word-vector steering from the product. The dials below are the only steering left.

---

## 4. Steering, second attempt: weight-space programming

### 4.1 The harness: an RL-style environment for an agent that programs weights

`steer/wsp/lab.py` holds LFM3-1.7B and exposes it to an **LLM programmer agent**. The agent gets a plain-English task (`steer/wsp/tasks/<axis>.md`) and three kinds of action:

| Action | What it does | Cost |
|---|---|---|
| `kernel` | Runs Python in a persistent kernel that can **read** the weights, earlier activations and generations. The model object is not in the kernel, so the agent cannot run the network there. | free |
| `forward` | One budgeted pass: either a batch of generations (base and candidate dials, with paired seeds), or a teacher-forced read of up to 256 texts (log-probs and activations). | **1 of 15 passes**; a failed request is free |
| `submit_dial` | Writes the final dial to `moods/<concept>__<anti>.pt`. | ends the run |

The lab enforces these rules or checks them afterwards in the logs:

- No gradients, autograd, optimisers or iterative fitting.
- The embedding, the output head and the final norm are off limits.
- α = 0 must be exactly the original model.

Every call is logged to `wsp_runs/<axis>/tool_calls.jsonl`, together with the agent's notes and final report. We call it "RL-style" because the agent acts under a budget and is judged on a held-out score. Nothing is trained: there is no policy update and no reward signal.

### 4.2 What the agent delivers: a dial

A dial is one low-rank edit per chosen layer that the wheel can turn per token:

$$W_\ell' = W_\ell + \alpha_t\, A_\ell B_\ell^{\top}, \qquad \alpha_t \in [0, 1]$$

It is applied with a forward hook, `y += α·(x·B)·Aᵀ`, so α can change on every token, and α = 0 is exactly the base model. All four shipped dials converged on the same closed-form recipe, which the pilot found first:

- **Direction.** $u_\ell$ is the mean residual at layer ℓ on mood text minus the opposite text, read teacher-forced after the *same bare opening*. Using the same opening means the direction comes from the continuation, not the prompt.
- **Write vector.** $A_\ell = c_\ell\,\lVert \bar h_\ell \rVert\,\mathrm{normalize}(g_\ell \odot u_\ell)$, where $g_\ell$ is the post-FFN norm gain. This is the direction in the MLP's output space that moves the residual most along $u_\ell$.
- **Gate.** $B_\ell = \bar x_\ell / \lVert \bar x_\ell \rVert^2$, where $\bar x_\ell$ is the mean input to the down-projection, so $x \cdot B \approx 1$ on typical text. The edit acts as a constant push in every context. The pilot deliberately avoided the paper's *directional gain*: a gain amplifies what neurons already write, so it would push calm text further toward calm.
- **Placement.** The MLP down-projections (`feed_forward.w2`) of layers 7, 9, 11, 13, 15 and 17. The strength is $c = 0.3$ on the early three and $0.1$ on the late three ("earlyW"). Each per-layer edit changes the weight matrix by about 0.25–1.2% of its norm.

### 4.3 The pilot: tense ↔ calm (local M4, 15 of 15 passes, 36 min)

The first agent worked from 48 hand-written contrast pairs. Its key results, from `wsp_runs/tense-calm/report.md`, on 16 held-out matched pairs:

| setting | Δ log p(tense) − log p(calm), nats/token | pairs moved the right way |
|---|---|---|
| dial, α = 0.25 | +0.30 | 16/16 |
| dial, α = 0.5 | +0.63 | 16/16 |
| dial, α = 1 | +1.19 | 16/16 |
| **random direction, same norm, α = 1** | **+0.02** | **11/16 (chance)** |
| sign-flipped dial, α = 1 | −0.97 | 0/16 |

The effect rises smoothly with α, the random edit does nothing, and flipping the sign reverses the effect. In generation (48 held-out pairs at α = 1): 30 samples were tenser and fluent, 2 were tenser but broken, 13 were unchanged and 3 were less tense. Most unchanged samples were prompts the base model had already turned into a maths word problem. The dial does not pull the model out of that mode.

### 4.4 Learning from the pilot: the task spec got stricter

The pilot's contrast pairs were written by hand, and the dial picked up some of their author's vocabulary. The task template (`steer/wsp/tasks/TEMPLATE.md`) for the remaining axes therefore required:

1. **Contrast data from the model itself.** Elicit mood and anti-mood continuations from LFM3 using raw-text genre headers, then *strip the header* and read each continuation after the bare opening.
2. **Filtering.** Drop samples that contain the mood words or their stems, samples in quiz or maths mode, and broken text. Keep pairs matched by opening, and report how many survive. The tender run, for example, kept 106 of 128 pairs.
3. **A vocabulary-leak check.** Project the edit through the final norm and the tied embedding, and rank all 128k tokens. On the tender axis the model's own "tender register" direction was aligned with the *word* " tender" (rank 5), even though no sample contained it. A closed-form projection (one QR per layer) removed that component. After the fix, no mood token was in the top 2,000.
4. **A norm-matched random control and held-out prompts.** The agent reads the samples itself.

### 4.5 Scaling out on Modal

Each remaining axis (tender ↔ cold, playful ↔ grave, lush ↔ plain) got its own lab on a Modal **L40S** (`steer/wsp/modal_lab.py`, `scripts/launch-axes.sh`). Three agents worked in parallel, one per lab. All three labs started in the same minute and **finished in 5–9 minutes** each (against 36 minutes for the local pilot). The longest single pass took 42.5 s. The agents used **6, 9 and 5** of their 15 passes.

Running agents in parallel exposed two harness bugs, both now fixed in `lab.py`:

- **Concurrent passes overwrote each other.** Two simultaneous `forward` calls got the same pass number, and the second overwrote the first. This cost the tense–calm v2 run half of its contrast data. The fix is a lock around forward and kernel calls.
- **`submit_dial` crashed on Modal** with a path error after writing the dial file. The dials were intact. The fix is a guard around the path bookkeeping.

### 4.6 A second tense dial, and why we kept the first

The v2 agent rebuilt tense ↔ calm with the stricter spec and scaled its dial to half strength, because the full-strength version broke 6% of samples. Head to head on 12 held-out prompts × 2 seeds (`dial_stats.py`, samples out of 24):

| α | arm | threat words | calm words | says "tense" | quiz mode |
|---|---|---|---|---|---|
| 1 | base | 0 | 6 | 0 | 5 |
| 1 | **v1 (shipped)** | **9** | 0 | 0 | 2 |
| 1 | v2 | 2 | 0 | 0 | 6 |
| 0.5 | v1 | 3 | 1 | 0 | 5 |
| 0.5 | v2 | 2 | 0 | 0 | 6 |

v2 was too weak at full scale, so v1 shipped. v1 is the strongest dial but also the least subtle: past α ≈ 0.65 it tips into melodrama and repetition. That is why the wheel marks 0.65 as a red line.

### 4.7 What the four dials do

Lexical hits are matches of each dial's own word list (in `dial_stats.py`); 24 samples per cell:

| dial | base | α = 0.6 | α = 1 | says its mood word at α = 1 |
|---|---|---|---|---|
| tender | 4 | 13 | 22 | 1/24 |
| lush | 1 | 12 | 28 | 0/24 |
| playful | 9 | 4 | 4 | 0/24 |

By word count, **playful looks like a failure**. Reading the samples says otherwise: playful text *is* lighter, but it gets there through structure, not vocabulary. Structural features on the same samples:

| setting | ?/! per sample | contractions per sample | words per sentence | mean word length |
|---|---|---|---|---|
| base | 0.46 | 0.29 | 11.0 | 4.08 |
| playful 0.6 | **1.04** | **0.54** | 10.3 | 4.07 |
| tender 1 | 0.50 | **0.75** | 10.0 | 3.99 |
| lush 1 | 0.12 | 0.29 | **12.5** | **4.39** |

Playful doubles the questions and exclamations and adds contractions. Lush writes longer sentences with longer words. This repeats the lesson from the orthography paper from the other side: **a word-list metric misses changes in register**. Counting mood words is not enough; you also have to measure structure, and read the text.

---

## 5. Detection and import

**Pangram** (`steer/pangram.py`). Each decision node is scored by Pangram's AI-detection API. A check runs 2.5 s after editing stops, results are cached by text, and documents under 20 words are skipped. Pangram's per-segment windows tint the text (red for AI, amber for AI-assisted). Each tree node sits in a blurred cloud whose colour mixes green (human), amber (assisted) and red (AI) in Pangram's proportions. Over time the tree becomes a map of which edits moved the document toward "human".

**Reducto** (`steer/reducto.py`). An uploaded PDF is parsed by Reducto into ordered blocks, merged into chunks, and imported as a **linear chain** of nodes: one node per chunk, with no branches. One Pangram scan of the whole document gives each chunk its own score. Clicking back along the chain reuses that scan instead of calling Pangram again, so imported scores stay stable.

---

## 6. The experiment: 100% AI → 48% AI

**Material.** A paper written entirely by an AI system: Sakana AI's *AI Scientist* sample "Adaptive Dual-Scale Denoising" (`samples/ai_scientist_adaptive_dual_scale_denoising.pdf`). Pangram scores the full import at 77% AI. The first chunk, the abstract, scores **100% AI**. As a control, *Attention Is All You Need* (2017, before modern LLMs) imports at **0% AI**, all 93 chunks human.

**Procedure.** Starting from the abstract, one person (Kojin) co-wrote forward in Cosign: mostly short `⌥Tab` model writes, each steered by the wheel, plus a few typed edits. Many branches were tried, and the tree keeps them all. The branch reported here is the one in the shipped save file. Running `python3 scripts/branch_stats.py` prints:

```
node      n162  (482 words)
start     1 imported chunk(s); chunk score 100% AI
decisions 43: 3 type, 40 opt-tab
model writes (opt-tab) 40, steered with the wheel 39
moods     plain 11, tender 8, playful 8, calm 8, grave 7, cold 5, tense 4, lush 3
strength  min 0.02  median 0.11  max 0.65  (redline 0.65)
authorship doc 48%, model 46%, human 6%
Pangram AI after each decision: first 0.89 -> last 0.48 (36 scored)
```

**What happened.** Pangram's AI score fell almost steadily, from 0.89 after the first typed edit to **0.48**. Along the way the writer **changed mood constantly**: 39 of 40 model writes were steered, across all eight moods, mostly at *light* settings (median strength 0.11; only the maximum reached the red line). The finished text is 48% the original AI abstract, 46% steered model text and 6% typed. In other words, Pangram rated a document that is mostly machine-written as more than half human.

This is the hypothesis in action: per-sentence variation in mood, supplied by a human through the dials, produced text the detector did not read as monotone AI prose. A sibling branch in the same tree (node `n166`, 41 model writes) also ended at 48%.

To inspect it yourself, open `samples/ai_scientist_branch_48pct.cosign.json` in Cosign ("open a file…"). Every node keeps its text, its dial settings and its Pangram score.

---

## 7. Limitations

We state these as plainly as we can, because the result is easy to over-read.

- **Pangram may not know LFM3.** We do not know whether Pangram's training data includes text from LFM3 models. If it does not, a low AI score can show a gap in the detector, not text that reads as human. Part of the 48% could then come from the detector's training data, not from the dials.
- **No unsteered baseline yet.** The decisive test is to write the same branch with the dials off and compare the Pangram curves. That separates "steered LFM3 text reads as human" from "any LFM3 text reads as human". This is the next experiment.
- **One branch, one writer, one detector.** This is a demonstration, not a controlled study.
- **The dials are rough.**
  - Tense tips into melodrama above about 0.65.
  - About a quarter of prompts send the base model into quiz mode, and no dial fixes that.
  - Playful's effect is structural and modest.
  - The lexical and structural counts in `dial_stats.py` are crude proxies. The agents' per-sample judgements were unblinded, by one judge.
- **The anti side of each dial is just the base model.** α = 0 is the base model, not "extra calm". In the pilot, the sign-flipped dial suppressed tension more than it added calm.

---

## 8. How this was built

Cosign was built during the hackathon by one person working with AI agents. The division of labour was itself part of the method:

- **Direction and judgement (human).** Kojin set the hypothesis, the product, every interaction (key bindings, the wheel, the tree), and the experiment. Several key findings came from reading the output, not from metrics. One example: noticing that playful text was playful *in its structure*, which led to the structural measurements in section 4.7.
- **Implementation (coding agent).** A coding agent ([Hermes Agent](https://hermes-agent.nousresearch.com)) wrote the engine, the server, the editor, the lab and the analysis scripts under that direction. Each feature was checked by driving the real UI in a browser before it was reported done.
- **Weight-space programming (LLM programmer agents).** Each dial was written by a separate agent working inside the lab, under the rules in section 4.1. Its notes, tool logs and report are in `wsp_runs/<axis>/`. The agents chose the edits themselves. Humans wrote the task specs and the evaluation, and decided what shipped.
- **Verification habits that mattered:**
  - A norm-matched random control for every dial.
  - Held-out prompts only.
  - Clearing steering before resetting the prompt. An early comparison was contaminated by leftover steering in the prompt's cache; its files are kept and marked `_CONTAMINATED`.
  - A vocabulary-leak check through the tied embedding.
  - Recomputing every published number from saved files (`scripts/`).

---

## 9. Reproduce

```bash
uv sync                                        # Python 3.12, torch, transformers
cd frontend && pnpm install && pnpm build && cd ..
cp .env.example .env                           # optional keys: Pangram, Reducto, Modal
uv run cosign-web                              # http://127.0.0.1:8000

python3 scripts/dial_stats.py                  # section 3–4 tables (no model needed)
python3 scripts/branch_stats.py                # section 6 numbers (no model needed)

uv run wsp-lab --run my-axis --budget 15       # a weight-space programming lab, locally
uv run wsp status | kernel | forward           # the programmer's CLI
scripts/launch-axes.sh tender:cold lush:plain  # one Modal L40S lab per axis (needs Modal credentials in .env)
```
