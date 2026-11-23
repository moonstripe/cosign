# Pre-registration: does per-sentence mood steering change Pangram's verdict on LFM3 text?

Written before any generation or scoring for this experiment. Motivated by issues #1, #2, #3 and #5.
The original headline (100% → 48% AI) turned out to be dilution: the score equalled the share of
imported AI text, and every appended character was labelled human regardless of steering (#1). This
experiment scores the **generated text alone**, so it cannot be diluted.

## Hypothesis

H1: LFM3-1.7B continuations whose mood changes **sentence by sentence** (via the weight-space dials) get a
**lower Pangram AI score** than (a) unsteered continuations and (b) continuations held at one constant mood,
at matched length.

## Arms (same openings, same seeds)

| arm | what writes the continuation |
|---|---|
| A `unsteered` | LFM3-1.7B raw continuation, dials off (α = 0) |
| B `constant` | one positive-side mood for the whole sample (tense, tender, playful or lush, rotated over samples); strength redrawn every sentence |
| C `vary_pos` | a new positive-side mood every sentence (uniform over the 4, no immediate repeat); same strength draws as B |
| D `vary_both` | a new mood every sentence from all 8 wedges, incl. the anti side (negative α, untested, see #3); same strength draws |
| E `assistant` | LFM3-1.7B with its chat template ("Continue the following text…", thinking off), dials off: the assistant persona |
| H `human` | human-written text, no generation (5 academic slices from *Attention Is All You Need* 2017, 5 passages from pre-1900 novels on Project Gutenberg) |

Strengths for B/C/D are drawn per sentence from U(0.25, 0.60): clearly audible, below the 0.65 red line.
(Note: the original session used lighter settings, median 0.11.) Sampling: temperature 0.8, top-k 50,
repetition penalty 1.1 (the editor's defaults). Per sample the token RNG is seeded identically across arms.

## Openings

10 openings × 2 seeds = 20 samples per generated arm.
- 5 academic: chunks 1, 2, 12, 23 and 58 of the AI-Scientist paper (abstract, introduction, related work, background, results).
- 5 narrative: the held-out openings in `steer/wsp/compare.py` (kettle, Sunday mornings, library, orchard, Mara).

## Measurement

Each sample is generated to about 520 words and cut at the first sentence end at or after **150, 300 and
500 words**. Each cut is scored by Pangram (`pangram-4`) **on its own**, with no opening included.
Pre-specified primary metric: `fraction_ai` (0–1). Also recorded: the headline label, windows,
`fraction_ai_assisted`, quiz-mode flag and repetition flag.

Calls: 5 generated arms × 20 samples × 3 lengths + 10 human × 3 lengths = 330.

## Analysis (fixed in advance)

- Primary comparisons, paired by (opening, seed): **C − A** and **C − B**, at 300 and 500 words.
- Report the mean difference with a 95% bootstrap CI (10,000 resamples over the 20 pairs) and the count of pairs where C < the comparison arm.
- **H1 is supported** only if both C − A and C − B are negative with CIs excluding 0 at **both** 300 and 500 words.
  If only some of these hold, report "mixed". If none hold, report a negative result.
- Secondary (descriptive, no claim): D vs A (negative α), E vs A (persona), H (detector calibration), per-register splits (academic vs narrative), and the length trend.
- Samples are not filtered. Quiz-mode and repetition rates are reported per arm.

## What would change the write-up

Whatever the outcome, the README/METHODOLOGY headline is rewritten to report the appended-text-only result
from this experiment, and the 48% figure is retired as a headline.
