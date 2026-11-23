# DRAFT pre-registration: is the context flip about LFM3, or about Pangram's segmentation?

*Status: draft, not yet run. No data has been generated or scored for this test.*

## Background

- In the hackathon demo document, ~1,800 characters of appended text (89% LFM3, steered) were labelled **human (0.0)**, placed after an AI-flagged abstract.
- Scored **alone**, that same text is labelled **AI (1.0)** (`experiments/detector/n162_appended_alone.json`). Re-scoring the whole document today reproduces the 0.0 (Pangram v4.1, same windows).
- Unsteered LFM3 text scored alone is flagged at 0.82–0.91 (`experiments/detector/results.md`).

So "Pangram does not recognise LFM3 at all" is already unlikely. Two explanations remain for the flip:

- **H-LFM3:** in context, Pangram misses *LFM3* text specifically (or steered LFM3 specifically); other AI text in the same position would still be flagged.
- **H-SEG:** the flip is a property of Pangram's segmentation: *any* AI text placed after an AI-flagged segment tends to be labelled human, whoever wrote it.

## Design: appended-text source × context, scored as whole documents

**Context (what comes before the appended text):**

| code | prefix |
|---|---|
| `none` | nothing: the appended text scored alone |
| `ai` | the AI Scientist abstract (scores 1.0 AI alone; the demo's prefix) |
| `human` | the *Attention Is All You Need* abstract (scores 0.0 alone) |

**Appended text source (each cut at ~250 words, the demo's length, and ~500 words):**

| code | text | why |
|---|---|---|
| `lfm3` | LFM3-1.7B, unsteered, continuing the prefix | is it LFM3 itself? |
| `lfm3_steer` | LFM3-1.7B, replaying the demo's recorded steering schedule (n162: 40 writes, same sizes, moods, strengths; no human edits) | does the steering add anything? |
| `other_ai` | AI text from a different model family that Pangram flags alone: later sections of the AI Scientist paper (body chunks, not the abstract) | positive control: if this also flips, it is not about LFM3 |
| `human` | human text: passages from the 2017 Attention paper body | negative control: should stay human everywhere |

*For `ai` and `human` contexts with `lfm3`/`lfm3_steer`, the continuation is generated from that prefix. `other_ai` and `human` sources are fixed texts placed after each prefix.*

**Units:** 6 items per source (6 openings or passages; LFM3 arms use 6 seeds from the real abstract plus 6 from the Attention abstract).

**Measurement.** For each document, take Pangram's windows (v4.1) and compute the **share of appended characters that fall in AI-labelled windows**. A character in a window that straddles the boundary counts by the window's label. Also recorded: the whole-document score, and the window boundaries.

**Calls:** 4 sources × 3 contexts × 2 lengths × 6 items = 144. Some `none` cells overlap with existing scores and can be reused.

## Predictions (written before data)

| if … | then … |
|---|---|
| `other_ai` after an `ai` prefix is also mostly labelled human | **H-SEG**: the flip is a segmentation effect, not about LFM3. |
| `other_ai` stays flagged after an `ai` prefix, but `lfm3` flips | **H-LFM3**: Pangram misses LFM3 in context specifically. |
| `lfm3_steer` flips clearly more than `lfm3` in the `ai` context | the steering contributes to evading the detector, in context. |
| the flip appears after an `ai` prefix but not a `human` prefix | the effect depends on a preceding AI-flagged segment (a contrast or boundary effect). |
| `human` text is ever flagged | detector calibration problem; note before interpreting anything else. |

## Primary analysis

- **Primary:** the appended AI share in the `ai` context, compared across sources, each against its own `none` score (the flip size = alone minus in context), with 95% bootstrap CIs over items.
- **Steering question:** `lfm3_steer` vs `lfm3` flip size in the `ai` context, paired by seed.
- No re-cutting after data is seen: lengths, items, and the metric are fixed here.

## Open choices before running

1. Whether `other_ai` should also include a frontier chat model's text (written fresh), in addition to the AI Scientist body chunks.
2. Whether to add an intermediate prefix length (the demo's prefix was ~1,700 characters, about half the document).
3. Whether to add a `lfm3_steer` variant with positive α only (the demo used negative α in 31 of 54 settings).
