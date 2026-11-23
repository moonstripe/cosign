## tense <-> calm: word vector vs dial (samples containing ..., out of n)
strength = signpost strength (10 = full = dial alpha 1)

| strength | arm  | n  | grammar talk | says 'tense' | threat words | quiz mode |
|---|---|---|---|---|---|---|
| 4 | base | 24 | 0 | 0 | 0 | 4 |
| 4 | dial | 24 | 1 | 0 | 1 | 5 |
| 4 | word | 24 | 1 | 0 | 1 | 3 |
| 7 | base | 24 | 0 | 0 | 0 | 4 |
| 7 | dial | 24 | 0 | 0 | 1 | 5 |
| 7 | word | 24 | 13 | 7 | 1 | 2 |
| 10 | base | 24 | 0 | 0 | 0 | 4 |
| 10 | dial | 24 | 0 | 0 | 5 | 1 |
| 10 | word | 24 | 20 | 13 | 0 | 1 |

## tense <-> calm: v1 (earlyW, shipped) vs v2 (tc2_final), samples containing ..., out of 12 prompts x 2 seeds

| alpha | arm | threat words | calm words | says 'tense' | quiz mode |
|---|---|---|---|---|---|
| 0.5 | base | 0 | 6 | 0 | 5 |
| 0.5 | v1 | 3 | 1 | 0 | 5 |
| 0.5 | v2 | 2 | 0 | 0 | 6 |
| 1 | base | 0 | 6 | 0 | 5 |
| 1 | v1 | 9 | 0 | 0 | 2 |
| 1 | v2 | 2 | 0 | 0 | 6 |

## all dials vs base (12 held-out prompts x 2 seeds = 24 samples per cell)

Lexicon hits = total matches of the dial's own word list (see LEX in this file).

| dial | lexicon hits: base | alpha 0.6 | alpha 1 | mood word said (alpha 1) |
|---|---|---|---|---|
| tender | 4 | 13 | 22 | 1/24 |
| playful | 9 | 4 | 4 | 0/24 |
| lush | 1 | 12 | 28 | 0/24 |

Structural features (same samples):

| setting | ?/! per sample | contractions per sample | hedges per sample | words per sentence | mean word length | quiz mode (samples) |
|---|---|---|---|---|---|---|
| base | 0.46 | 0.29 | 0.25 | 11.02 | 4.08 | 5 |
| tender 0.6 | 0.75 | 0.58 | 0.38 | 11.62 | 3.97 | 7 |
| tender 1 | 0.50 | 0.75 | 0.33 | 9.96 | 3.99 | 4 |
| playful 0.6 | 1.04 | 0.54 | 0.29 | 10.28 | 4.07 | 5 |
| playful 1 | 1.00 | 0.50 | 0.29 | 9.77 | 3.97 | 3 |
| lush 0.6 | 0.42 | 0.25 | 0.38 | 11.00 | 4.24 | 0 |
| lush 1 | 0.12 | 0.29 | 0.12 | 12.52 | 4.39 | 1 |

