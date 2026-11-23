# Detector experiment: results

Pre-registered in `PREREG.md`. Pangram `fraction_ai` of the **generated text alone** (no opening).

## Mean Pangram AI score by arm and length

| arm | 150 words | 300 words | 500 words | quiz-mode samples | repetitive samples |
|---|---|---|---|---|---|
| unsteered | 0.82 (n=20, 17 AI-majority) | 0.89 (n=17, 15 AI-majority) | 0.91 (n=11, 10 AI-majority) | 4/20 | 2/20 |
| constant | 0.84 (n=20, 17 AI-majority) | 0.92 (n=17, 16 AI-majority) | 0.99 (n=13, 13 AI-majority) | 9/20 | 3/20 |
| vary_pos | 0.71 (n=20, 15 AI-majority) | 0.87 (n=15, 14 AI-majority) | 0.86 (n=13, 11 AI-majority) | 8/20 | 4/20 |
| vary_both | 0.77 (n=20, 16 AI-majority) | 0.98 (n=19, 19 AI-majority) | 0.97 (n=17, 17 AI-majority) | 9/20 | 2/20 |
| assistant | 1.00 (n=19, 19 AI-majority) | 1.00 (n=19, 19 AI-majority) | 1.00 (n=17, 17 AI-majority) | 3/20 | 0/20 |
| human | 0.00 (n=10, 0 AI-majority) | 0.00 (n=10, 0 AI-majority) | 0.00 (n=10, 0 AI-majority) | 0/10 | 0/10 |

## Paired differences (row arm minus column arm), mean [95% bootstrap CI], pairs lower / higher

| comparison | kind | 150 words | 300 words | 500 words |
|---|---|---|---|---|
| vary_pos − unsteered | primary | -0.109 [-0.317, +0.102] 7/6 of 20 | -0.058 [-0.280, +0.173] 5/1 of 13 | -0.125 [-0.375, +0.000] 1/0 of 8 |
| vary_pos − constant | primary | -0.124 [-0.364, +0.117] 7/7 of 20 | -0.040 [-0.250, +0.178] 5/3 of 14 | -0.063 [-0.186, +0.013] 2/2 of 10 |
| vary_both − unsteered | secondary | -0.049 [-0.277, +0.186] 7/7 of 20 | +0.043 [-0.024, +0.152] 1/4 of 16 | -0.031 [-0.092, +0.000] 2/0 of 10 |
| constant − unsteered | secondary | +0.015 [-0.173, +0.196] 6/7 of 20 | +0.050 [-0.042, +0.193] 2/4 of 16 | -0.008 [-0.020, +0.000] 2/0 of 9 |
| assistant − unsteered | secondary | +0.136 [+0.040, +0.264] 0/8 of 19 | +0.072 [+0.000, +0.205] 0/4 of 16 | +0.091 [+0.000, +0.273] 0/1 of 11 |

### academic openings only

| comparison | 150 words | 300 words | 500 words |
|---|---|---|---|
| vary_pos − unsteered | -0.027 [-0.223, +0.169] 3/2 of 10 | -0.047 [-0.112, +0.000] 2/0 of 8 | +0.000 [+0.000, +0.000] 0/0 of 5 |
| vary_pos − constant | -0.017 [-0.314, +0.283] 3/4 of 10 | +0.069 [-0.120, +0.336] 2/2 of 9 | -0.074 [-0.248, +0.025] 1/1 of 7 |
| vary_both − unsteered | -0.123 [-0.406, +0.141] 3/3 of 10 | +0.003 [+0.000, +0.009] 0/2 of 9 | -0.000 [-0.000, +0.000] 1/0 of 6 |
| constant − unsteered | -0.010 [-0.268, +0.211] 4/3 of 10 | +0.001 [-0.045, +0.048] 1/2 of 9 | -0.010 [-0.029, +0.000] 1/0 of 6 |
| assistant − unsteered | +0.116 [+0.009, +0.251] 0/4 of 10 | +0.016 [+0.000, +0.048] 0/2 of 9 | +0.000 [+0.000, +0.000] 0/0 of 6 |

### narrative openings only

| comparison | 150 words | 300 words | 500 words |
|---|---|---|---|
| vary_pos − unsteered | -0.191 [-0.539, +0.194] 4/4 of 10 | -0.075 [-0.626, +0.525] 3/1 of 5 | -0.333 [-1.000, +0.000] 1/0 of 3 |
| vary_pos − constant | -0.230 [-0.597, +0.167] 4/3 of 10 | -0.234 [-0.634, +0.082] 3/1 of 5 | -0.036 [-0.121, +0.012] 1/1 of 3 |
| vary_both − unsteered | +0.024 [-0.319, +0.398] 4/4 of 10 | +0.095 [-0.058, +0.344] 1/2 of 7 | -0.076 [-0.229, +0.000] 1/0 of 4 |
| constant − unsteered | +0.039 [-0.261, +0.325] 2/4 of 10 | +0.113 [-0.088, +0.429] 1/2 of 7 | -0.004 [-0.012, +0.000] 1/0 of 3 |
| assistant − unsteered | +0.158 [+0.018, +0.381] 0/4 of 9 | +0.143 [+0.000, +0.429] 0/2 of 7 | +0.200 [+0.000, +0.600] 0/1 of 5 |

## Pre-registered verdict on H1

**NOT SUPPORTED** (criterion: C − A and C − B both have 95% CIs below 0 at 300 and 500 words; vary_pos−unsteered@300: no, vary_pos−unsteered@500: no, vary_pos−constant@300: no, vary_pos−constant@500: no)
