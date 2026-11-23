# Methodology

> **Status (23 November 2026): research in progress; the headline result is retracted.**
> The hackathon version of this document said that Cosign took an AI-written paper "from 100% AI to 48% AI" and that the dials caused it. An external review showed that the drop was dilution: Pangram's score tracked how much of the document was still the original AI text, and it labelled every appended character human ([#1](https://github.com/moonstripe/cosign/issues/1), [#2](https://github.com/moonstripe/cosign/issues/2)). A pre-registered follow-up (§7) scored the generated text on its own and found **no reliable effect of the dials on Pangram**. Section 6 keeps the original demo, marked retracted, so the record stays complete. The [corrections log](#corrections-log) lists every change from the hackathon version.

Cosign won **first place at the Non Slop Hackathon** (Pangram, Nous Research, Reducto, Modal; 22 November 2026). This document explains how the system was built, what each part was tested against, and which results hold up. It also covers what we do not know yet.

Every number here can be traced to a file in this repo:

| Source | What it holds |
|---|---|
| `wsp_runs/<axis>/report.md`, `notes.md`, `tool_calls.jsonl` | the weight-space programmers' own reports, notes and complete tool logs |
| `scripts/dial_stats.py` → `wsp_runs/dial_stats.md` | every comparison table in sections 3 and 4 |
| `scripts/branch_stats.py` → reads `samples/ai_scientist_branch_48pct.cosign.json` | every number in section 6 |
| `experiments/detector/` (`PREREG.md`, `run.py`, `samples.jsonl`, `scores.jsonl`, `results.md`) | every number in section 7 |

---

## Corrections log

**23 November 2026**, after an external review ([issues #1–#14](https://github.com/moonstripe/cosign/issues)). Every claim below was checked against the committed files before it was corrected.

| Hackathon version said | Correction | Issue |
|---|---|---|
| Cosign took an AI-written paper from 100% to 48% AI; per-sentence mood variation "produced text the detector did not read as monotone AI prose" | **Retracted.** The score equalled the share of original AI text (1,706 / 3,527 characters = 0.484). The appended text (89% LFM3) was labelled human in the document, but scores 1.0 AI on its own. A controlled test found no reliable dial effect (§6, §7). | #1, #5 |
| The score "fell almost steadily" to 48%; a sibling branch "also ended at 48%" | 0.477 is the **lowest score in the whole tree**. Neither node is an end point: their continuations score 0.60–0.73. | #2 |
| "The anti side of each dial is just the base model"; α ∈ [0, 1] | The anti wedges (calm, cold, grave, plain) run the dial at **negative α**, which was tested only for tense ↔ calm. 31 of the 54 dial settings on the demo branch were negative. | #3 |
| §4.7: tender and lush move their own word lists; playful works "through structure" | The lush dial hits the tender list about as often as the tender dial does (26 vs 22 at α = 1), and tender adds more contractions than playful at matched strength. The tables do not show mood-specific effects. | #4 |
| Word vectors failed and *weight-space* dials succeeded | The comparison changed the direction source and the mechanism at once. With x·B ≈ 1 the dial is close to a constant multi-layer steering vector. The credit goes to the direction source until an ablation shows otherwise. | #8 |
| The kernel "cannot run the network"; no gradients | Not enforced: the kernel runs in the same process as the model. A search of all five agents' tool logs finds no access to the model and no gradient calls, but the guarantee was procedural. | #9 |
| All four dials converged on one recipe (strong early layers) | True for tense, tender and lush; **playful uses the reverse** (weak early, strong late). | #11.1 |
| `submit_dial` writes the shipped dial | Three Modal dials were placed by hand after a crash. The v2 run overwrote the tense dial, and v1 was restored without a record. The shipped tense dial is tensor-identical to v1. | #11.2 |
| The full import scores 77% AI | Not in any committed file; dropped. The committed per-chunk scores average 0.90. | #11.3 |
| Each dial "had to beat a random edit" | Only inside each agent's own run: its own prompts, its own unblinded reading, one random draw. | #11.4 |
| "Pangram may not know LFM3" | Resolved: Pangram flags unsteered LFM3 text alone at 0.82–0.91 (§7). | — |

---

## 1. The hypothesis

> People change mood a little between sentences. Language models write in one voice.

AI text is easy to spot partly because it is *monotone*: one register, held from the first sentence to the last. The usual fix is to clean the text up afterwards, with paraphrasers and "humanizers". We wanted to test the opposite approach. Give the writer a control on the model's voice, per sentence, built into the model itself. Then let a human and the model write together.

*Status: the detector part of this hypothesis is not supported by the controlled test in §7. The per-sentence arm trended lower, but the trend was not distinguishable from noise.*

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

**Caveat ([#8](https://github.com/moonstripe/cosign/issues/8)).** This comparison changes two things at once: where the direction comes from (a word vs. the model's own mood-register text) and how it is applied (a residual add vs. a weight edit). Because the dials' gate gives x·B ≈ 1 on typical text (§4.2), a dial behaves almost like a constant steering vector added at six layers. So the table supports "a better direction source". It does not yet show that editing the weights adds anything. A 2 × 2 ablation is planned.

---

## 4. Steering, second attempt: weight-space programming

### 4.1 The harness: an RL-style environment for an agent that programs weights

`steer/wsp/lab.py` holds LFM3-1.7B and exposes it to an **LLM programmer agent**. The agent gets a plain-English task (`steer/wsp/tasks/<axis>.md`) and three kinds of action:

| Action | What it does | Cost |
|---|---|---|
| `kernel` | Runs Python in a persistent kernel that can **read** the weights, earlier activations and generations. The model object is not placed in the kernel's namespace. | free |
| `forward` | One budgeted pass: either a batch of generations (base and candidate dials, with paired seeds), or a teacher-forced read of up to 256 texts (log-probs and activations). | **1 of 15 passes**; a failed request is free |
| `submit_dial` | Writes the final dial to `moods/<concept>__<anti>.pt`. | ends the run |

The rules:

- No gradients, autograd, optimisers or iterative fitting.
- The embedding, the output head and the final norm are off limits.
- α = 0 must be exactly the original model.

**These rules were stated and audited, not enforced ([#9](https://github.com/moonstripe/cosign/issues/9)).** The kernel is a plain `exec` in the same process as the model, so an agent could have reached the model, run free passes or used gradients. We searched all five agents' tool logs for model access, gradient calls and budget tampering and found none. That is evidence of compliance, not a guarantee. Sandboxing the kernel is an open fix.

Every call is logged to `wsp_runs/<axis>/tool_calls.jsonl`, together with the agent's notes and final report. We call it "RL-style" because the agent acts under a budget and is judged on a held-out score. Nothing is trained: there is no policy update and no reward signal.

### 4.2 What the agent delivers: a dial

A dial is one low-rank edit per chosen layer that the wheel can turn per token:

$$W_\ell' = W_\ell + \alpha_t\, A_\ell B_\ell^{\top}, \qquad \alpha_t \in [-1, 1]$$

It is applied with a forward hook, `y += α·(x·B)·Aᵀ`, so α can change on every token, and α = 0 is exactly the base model. Positive α is the axis's first mood (tense, tender, playful, lush). **Negative α is what the wheel's second mood (calm, cold, grave, plain) does.** Negative α was evaluated only for tense ↔ calm, and only at α = −1 in the pilot ([#3](https://github.com/moonstripe/cosign/issues/3)).

The dials share one closed-form recipe, which the pilot found first:

- **Direction.** $u_\ell$ is the mean residual at layer ℓ on mood text minus the opposite text, read teacher-forced after the *same bare opening*. Using the same opening means the direction comes from the continuation, not the prompt.
- **Write vector.** $A_\ell = c_\ell\,\lVert \bar h_\ell \rVert\,\mathrm{normalize}(g_\ell \odot u_\ell)$, where $g_\ell$ is the post-FFN norm gain. This is the direction in the MLP's output space that moves the residual most along $u_\ell$.
- **Gate.** $B_\ell = \bar x_\ell / \lVert \bar x_\ell \rVert^2$, where $\bar x_\ell$ is the mean input to the down-projection, so $x \cdot B \approx 1$ on typical text. The edit acts as a constant push in every context. The pilot deliberately avoided the paper's *directional gain*: a gain amplifies what neurons already write, so it would push calm text further toward calm.
- **Placement.** The MLP down-projections (`feed_forward.w2`) of layers 7, 9, 11, 13, 15 and 17. Tense, tender and lush put the larger strength on the early three layers ("earlyW": $c = 0.3$ early, $0.1$ late for tense). **Playful does the reverse** (0.05 early, 0.15 late); per-layer ‖A‖·‖B‖ in `moods/playful__grave.pt` rises from 0.09–0.16 on layers 7–11 to 0.29–0.38 on layers 13–17. Each per-layer edit changes the weight matrix by about 0.25–1.2% of its norm.

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

These checks ran inside the agent's own run: its own held-out pairs, its own unblinded reading of the generations, and a single random draw ([#11](https://github.com/moonstripe/cosign/issues/11)). The same applies to the random-edit checks of the other three agents. The repository-level comparisons in §4.6 and §4.7 have no random arm yet ([#7](https://github.com/moonstripe/cosign/issues/7)).

### 4.4 Learning from the pilot: the task spec got stricter

The pilot's contrast pairs were written by hand, and the dial picked up some of their author's vocabulary. The task template (`steer/wsp/tasks/TEMPLATE.md`) for the remaining axes therefore required:

1. **Contrast data from the model itself.** Elicit mood and anti-mood continuations from LFM3 using raw-text genre headers, then *strip the header* and read each continuation after the bare opening.
2. **Filtering.** Drop samples that contain the mood words or their stems, samples in quiz or maths mode, and broken text. Keep pairs matched by opening, and report how many survive. The tender run, for example, kept 106 of 128 pairs.
3. **A vocabulary-leak check.** Project the edit through the final norm and the tied embedding, and rank all 128k tokens. On the tender axis the model's own "tender register" direction was aligned with the *word* " tender" (rank 5), even though no sample contained it. A closed-form projection (one QR per layer) removed that component. After the fix, no mood token was in the top 2,000.
4. **A norm-matched random control and held-out prompts.** The agent reads the samples itself.

The template told the agents that α runs over [0, 1], so none of them evaluated the anti side ([#3](https://github.com/moonstripe/cosign/issues/3)).

### 4.5 Scaling out on Modal

Each remaining axis (tender ↔ cold, playful ↔ grave, lush ↔ plain) got its own lab on a Modal **L40S** (`steer/wsp/modal_lab.py`, `scripts/launch-axes.sh`). Three agents worked in parallel, one per lab. All three labs started in the same minute and **finished in 5–9 minutes** each (against 36 minutes for the local pilot). The longest single pass took 42.5 s. The agents used **6, 9 and 5** of their 15 passes.

Running agents in parallel exposed two harness bugs, both now fixed in `lab.py`:

- **Concurrent passes overwrote each other.** Two simultaneous `forward` calls got the same pass number, and the second overwrote the first. This cost the tense–calm v2 run half of its contrast data. The fix is a lock around forward and kernel calls.
- **`submit_dial` crashed on Modal** with a path error after writing the dial file. The dial files were intact, but the runs' `state.json` files record no submission, and the three Modal dials were copied into `moods/` by hand. There is no committed manifest linking each shipped file to its run ([#11](https://github.com/moonstripe/cosign/issues/11)).

The Modal lab also exposed its kernel, which runs arbitrary code, through a public tunnel with no authentication, and the tunnel URLs are in the committed task files ([#10](https://github.com/moonstripe/cosign/issues/10)). The labs are shut down; the fix is open.

### 4.6 A second tense dial, and why we kept the first

The v2 agent rebuilt tense ↔ calm with the stricter spec and scaled its dial to half strength, because the full-strength version broke 6% of samples. Head to head on 12 held-out prompts × 2 seeds (`dial_stats.py`, samples out of 24):

| α | arm | threat words | calm words | says "tense" | quiz mode |
|---|---|---|---|---|---|
| 1 | base | 0 | 6 | 0 | 5 |
| 1 | **v1 (shipped)** | **9** | 0 | 0 | 2 |
| 1 | v2 | 2 | 0 | 0 | 6 |
| 0.5 | v1 | 3 | 1 | 0 | 5 |
| 0.5 | v2 | 2 | 0 | 0 | 6 |

v2 was too weak at full scale, so v1 shipped. The v2 run's `submit_dial` had overwritten `moods/tense__calm.pt`. The file was restored to v1 by hand, with no record; we verified that the shipped file is tensor-identical to `wsp_runs/tense-calm/earlyW_v1.pt`. v1 is the strongest dial but also the least subtle: past α ≈ 0.65 it tips into melodrama and repetition. That is why the wheel marks 0.65 as a red line.

### 4.7 What the four dials do

*Revised ([#4](https://github.com/moonstripe/cosign/issues/4)).* The hackathon version compared each dial only against the base model and concluded that each moves its own mood. Comparing each dial against the **other** dials changes that picture. Off-target matrix from `wsp_runs/all-dials/compare.json`, using the word lists in `dial_stats.py` (total hits over 24 samples, α = 1):

| word list ↓ / text written by → | base | tender dial | playful dial | lush dial |
|---|---|---|---|---|
| tender | 4 | 22 | 3 | **26** |
| playful | 9 | 4 | 4 | 6 |
| lush | 1 | 6 | 1 | **28** |

- **Lush is the only dial specific to its own word list** (28 hits vs at most 6 from the other dials).
- **Tender is not clearly specific.** The lush dial hits the tender list about as often as the tender dial (26 vs 22). Part of that is overlap (`soft…` is in both lists); without it, tender leads 21 to 16, with base at 4. Tender and lush appear to share a "descriptive, emotive" component; their write vectors have cosine 0.79 at layer 7.
- **Playful does not move its word list** (9 → 4), and its structural shifts are not specific either. It raises questions and exclamations (0.46 → 1.04 per sample at α = 0.6), but tender raises them too (0.75 at α = 0.6), and tender adds more contractions than playful at both strengths (0.58 vs 0.54 at α = 0.6; 0.75 vs 0.50 at α = 1).
- Each cell is 24 samples from 12 prompts, with no uncertainty estimate. Differences like 9 vs 4 are within prompt-level noise.

The lesson from the orthography paper still applies: a word-list metric misses register changes, and structure has to be measured too. But neither measure, as run here, shows that each dial moves *its own* mood. Blind multi-judge evaluation with a random arm is the planned fix ([#7](https://github.com/moonstripe/cosign/issues/7)).

---

## 5. Detection and import

**Pangram** (`steer/pangram.py`). Each decision node is scored by Pangram's AI-detection API. A check runs 2.5 s after editing stops, results are cached by text, and documents under 20 words are skipped. Pangram's per-segment windows tint the text (red for AI, amber for AI-assisted). Each tree node sits in a blurred cloud whose colour mixes green (human), amber (assisted) and red (AI) in Pangram's proportions.

The score covers the **whole document**: it is the share of characters that fall in windows Pangram labels AI. Imported text therefore counts. Adding text that Pangram labels human lowers the score even if the AI-labelled part is untouched. This is what produced the result in §6. Each node stores only the aggregate score, not the windows, so a score cannot be traced to authorship spans after the fact ([#6](https://github.com/moonstripe/cosign/issues/6)).

**Reducto** (`steer/reducto.py`). An uploaded PDF is parsed by Reducto into ordered blocks, merged into chunks, and imported as a **linear chain** of nodes: one node per chunk, with no branches. One Pangram scan of the whole document gives each chunk its own score. Clicking back along the chain reuses that scan instead of calling Pangram again, so imported scores stay stable.

---

## 6. The hackathon demo: 100% AI → 48% AI (retracted)

> **Retracted as evidence for the dials.** The numbers below are correct as measurements, and are kept for the record. The interpretation we gave at the hackathon is wrong.

**Material.** A paper written entirely by an AI system: Sakana AI's *AI Scientist* sample "Adaptive Dual-Scale Denoising" (`samples/ai_scientist_adaptive_dual_scale_denoising.pdf`). Its 82 imported chunks score 0.90 AI on average (7 score below 0.6). The first chunk, the abstract, scores **100% AI**. As a control, *Attention Is All You Need* (2017, before modern LLMs) imports at **0% AI**: all 93 chunks human.

**Procedure.** Starting from the abstract, one person (Kojin) co-wrote forward in Cosign: mostly short `⌥Tab` model writes, each steered by the wheel, plus a few typed edits. Many branches were tried, and the tree keeps them all. Running `python3 scripts/branch_stats.py` on the shipped save prints:

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

**What we claimed (retracted).** "Pangram's AI score fell almost steadily … to 0.48. … per-sentence variation in mood, supplied by a human through the dials, produced text the detector did not read as monotone AI prose."

**What actually happened** ([#1](https://github.com/moonstripe/cosign/issues/1)). In the final scan, Pangram labelled characters 0–1,706 (the original abstract) AI with score 1.0, and every window after that, covering all the appended text, human with score 0.0. The document score is 1,706 / 3,527 = **0.484**. At every one of the 36 scored decisions on the branch, the score equals the share of characters that are still original abstract, to within 0.005 (mean gap 0.001). So:

- The curve measures **dilution**. Typed text, unsteered model text, or any other text Pangram labelled human would have drawn the same curve.
- The data cannot show that the dials did anything.
- The appended text was **89% LFM3** (1,620 of 1,830 characters; 210 are the typed edits). **Scored on its own, the same appended text is labelled AI-Generated, score 1.0, as one window** (`experiments/detector/n162_appended_alone.json`). In the document, Pangram cut it into five small windows (91–1,252 characters) after the AI abstract and labelled each one human. The text did not change; only its context did. So the dials did not make this text read as human. Why the context flips Pangram's verdict is not known: Pangram's segmentation is not documented.
- **Selection.** The writer saw the live Pangram colours on the tree while choosing which branches to continue. Even apart from the context effect, the branch was partly selected against the detector.

**Selection** ([#2](https://github.com/moonstripe/cosign/issues/2)). n162 (0.484) and its children n163 and n166 (both 0.477) are the **lowest scores in the whole tree** (95 scored nodes). They are not end points: their continuations score 0.60–0.73 (n164 0.73, n165 0.61, n167 0.60, n168 0.60, n169 0.73). Across the tree, the nodes that depart most from pure dilution sit about +0.25 above it, at around 490 words, where the model text is long enough for Pangram to flag a window inside it.

**Reversed dials.** 31 of the 54 dial settings on this branch (plain, calm, grave, cold) ran a dial at negative α, a mode with no evaluation for three of the four axes ([#3](https://github.com/moonstripe/cosign/issues/3)).

The save file is still in the repo (`samples/ai_scientist_branch_48pct.cosign.json`). Every node keeps its text, its dial settings and its Pangram score.

---

## 7. The controlled test: generated text scored alone

*Pre-registered: `experiments/detector/PREREG.md` was committed before any generation or scoring. Results: `experiments/detector/results.md`; every sample and every Pangram response is in `samples.jsonl` and `scores.jsonl`.*

**Question.** Does per-sentence mood steering lower Pangram's AI score for LFM3 text, compared with unsteered text and with one constant mood, when the generated text is scored on its own?

**Design.**

- **Openings.** 10 openings × 2 seeds: 5 chunks of the AI Scientist paper (abstract, introduction, related work, background, results) and 5 narrative openings.
- **Arms.** Every arm is generated from the same openings with the same token-sampling seed:
  - unsteered;
  - one positive mood held constant;
  - a new positive mood every sentence;
  - a new mood every sentence from all 8 wedges, including negative α;
  - LFM3 through its chat template ("Continue the following text…"), i.e. the assistant persona;
  - human text: five passages from the 2017 Attention paper and five from 19th-century novels.
- **Strengths.** Each sentence's strength is drawn from U(0.25, 0.60). The demo session used lighter settings (median 0.11).
- **Scoring.** Each sample is cut at the first sentence end at or after 150, 300 and 500 words. Each cut is scored by Pangram (v4.1) **alone**, with no opening attached. That made 287 calls.
- **Decision rule.** Fixed in advance: H1 holds only if "every sentence" beats both "unsteered" and "constant" at 300 and 500 words, with 95% bootstrap CIs below 0.

**Results.** Mean Pangram AI score (samples with a cut at that length):

| arm | 150 words | 300 words | 500 words |
|---|---|---|---|
| unsteered | 0.82 (20) | 0.89 (17) | 0.91 (11) |
| one mood, constant | 0.84 (20) | 0.92 (17) | 0.99 (13) |
| new mood every sentence | 0.71 (20) | 0.87 (15) | 0.86 (13) |
| every sentence, incl. negative α | 0.77 (20) | 0.98 (19) | 0.97 (17) |
| assistant persona | 1.00 (19) | 1.00 (19) | 1.00 (17) |
| human | 0.00 (10) | 0.00 (10) | 0.00 (10) |

Paired differences (same opening and seed), mean [95% bootstrap CI]:

| comparison | 150 words | 300 words | 500 words |
|---|---|---|---|
| every sentence − unsteered | −0.11 [−0.32, +0.10] | −0.06 [−0.28, +0.17] | −0.13 [−0.38, 0.00] |
| every sentence − constant | −0.12 [−0.36, +0.12] | −0.04 [−0.25, +0.18] | −0.06 [−0.19, +0.01] |
| assistant − unsteered | +0.14 [+0.04, +0.26] | +0.07 [0.00, +0.21] | +0.09 [0.00, +0.27] |

**Verdict: H1 is not supported.** The per-sentence arm is lower on average, but every CI includes 0.

What else the test shows:

- **Pangram does catch LFM3.** Unsteered text alone scores 0.82–0.91, and human text scores 0.00. This resolves the old limitation "Pangram may not know LFM3" in Pangram's favour.
- **Reversed dials do not help.** At 300 and 500 words, the arm that includes negative α scores at or above unsteered.
- **The assistant persona is the most detectable.** The no-persona design of §2 gives a small reduction, which is clearly distinguishable from noise at 150 words. Raw continuation is still flagged as AI.

**Limits of this test.**

- Many raw continuations hit end-of-text before 500 words, so the longer comparisons have only 8–14 matched pairs. The test cannot rule out a modest effect.
- Pangram is one detector, and its windowing is not documented.
- Quiz-mode drift affected 3–9 of 20 samples per arm. Samples were not filtered, as pre-registered.

---

## 8. Limitations and open questions

- **The per-sentence trend is unresolved.** A −0.04 to −0.13 difference is consistent with no effect and with a small one. A larger, separately pre-registered replication is the next step; this test will not be re-cut.
- **Why does context flip Pangram's verdict?** The demo's appended text scores 1.0 AI alone and 0.0 inside the document, after a strongly AI segment. Whether this is a general property of Pangram's segmentation (for example, AI text placed after an AI-flagged segment escapes detection) is untested. It may matter more than the dials did, and is worth reporting to Pangram. Persisting windows per node ([#6](https://github.com/moonstripe/cosign/issues/6)) would help trace it.
- **Reversed dials are mostly untested** ([#3](https://github.com/moonstripe/cosign/issues/3)). They should either be evaluated (dose–response, random control, held-out generations) or disabled.
- **Dial specificity is unproven** ([#4](https://github.com/moonstripe/cosign/issues/4), [#7](https://github.com/moonstripe/cosign/issues/7)). Word-list and structure counts on 24 samples are crude; judgements so far were unblinded, by one judge.
- **Weight edit vs steering vector** ([#8](https://github.com/moonstripe/cosign/issues/8)). The dials may add nothing beyond a well-chosen steering vector.
- **The dials are rough.**
  - Tense tips into melodrama above about 0.65.
  - About a quarter of prompts send the base model into quiz mode, and no dial fixes that.
- **Engineering and security** ([#9](https://github.com/moonstripe/cosign/issues/9), [#10](https://github.com/moonstripe/cosign/issues/10), [#12](https://github.com/moonstripe/cosign/issues/12)–[#14](https://github.com/moonstripe/cosign/issues/14)): the lab kernel is not sandboxed; the Modal lab exposed it publicly; the local server lacks Origin checks; there are WebSocket robustness bugs; and the Reducto client sends the API key to result URLs. Fixes are open.

---

## 9. How this was built

Cosign was built during the hackathon by one person working with AI agents. The division of labour was itself part of the method:

- **Direction and judgement (human).** Kojin set the hypothesis, the product, every interaction (key bindings, the wheel, the tree), and the experiment. Several findings came from reading the output, not from metrics, for example noticing structural changes in playful text (§4.7, since revised).
- **Implementation (coding agent).** A coding agent ([Hermes Agent](https://hermes-agent.nousresearch.com)) wrote the engine, the server, the editor, the lab and the analysis scripts under that direction. Each feature was checked by driving the real UI in a browser before it was reported done.
- **Weight-space programming (LLM programmer agents).** Each dial was written by a separate agent working inside the lab, under the rules in section 4.1. Its notes, tool logs and report are in `wsp_runs/<axis>/`. The agents chose the edits themselves. Humans wrote the task specs and the evaluation, and decided what shipped.
- **Review.** After the hackathon, an independent review agent audited the repository and filed issues #1–#14. Its two critical findings overturned the headline result. Every finding was checked against the committed files before this document was corrected.
- **Verification habits that mattered, and where they fell short:**
  - A norm-matched random control for every dial, but only inside each agent's run, with one draw.
  - Held-out prompts only.
  - Clearing steering before resetting the prompt. An early comparison was contaminated by leftover steering in the prompt's cache; its files are kept and marked `_CONTAMINATED`.
  - A vocabulary-leak check through the tied embedding.
  - Recomputing every published number from saved files (`scripts/`). This confirmed the numbers but not their interpretation. The demo's denominator (imported text) was the missing control.
  - Pre-registering the follow-up test before collecting data.

---

## 10. Acknowledgements

Thank you to [Pangram Labs](https://www.pangram.com), [Nous Research](https://nousresearch.com), [Reducto AI](https://reducto.ai) and [Modal](https://modal.com) for hosting the Non Slop Hackathon, and for the credits that made this work possible.

---

## 11. Reproduce

```bash
uv sync                                        # Python 3.12, torch, transformers
cd frontend && pnpm install && pnpm build && cd ..
cp .env.example .env                           # optional keys: Pangram, Reducto, Modal
uv run cosign-web                              # http://127.0.0.1:8000

python3 scripts/dial_stats.py                  # section 3–4 tables (no model needed)
python3 scripts/branch_stats.py                # section 6 numbers (no model needed)

uv run python experiments/detector/run.py report   # section 7 tables from the saved scores (no model, no API)
uv run python experiments/detector/run.py gen      # regenerate samples (model; ~1 h on an M4; needs the Gutenberg texts, see run.py)
uv run python experiments/detector/run.py score    # re-score with Pangram (~290 API calls)

uv run wsp-lab --run my-axis --budget 15       # a weight-space programming lab, locally
uv run wsp status | kernel | forward           # the programmer's CLI
scripts/launch-axes.sh tender:cold lush:plain  # one Modal L40S lab per axis (needs Modal credentials; see the security note in §4.5)
```
