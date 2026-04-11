# Can We Train Models to Solve Hard Sudokus?

*April 2026*

---

In [Post 1](/post1.html), I showed that masked diffusion models learn constraint structure better than autoregressive transformers on easy Sudoku puzzles. The position gradient disappeared, constraint violations dropped from 6.16 to zero with iterative decoding, and puzzle accuracy jumped from 12% to 100%. The architecture matters.

But easy Sudoku puzzles are a specific kind of problem. All 500,000 training puzzles had 31–36 given cells and were solvable by constraint propagation alone. No search required. The models learned to propagate constraints. They never learned to search.

When I tested them on hard puzzles (17–25 givens, requiring genuine backtracking) both models collapsed. The autoregressive model produced outputs with 25+ constraint violations per puzzle. The masked diffusion model with k=1 iterative decoding reduced this to 2.77 violations and solved 6 of 1000 hard puzzles correctly. Better, but not by much.

The obvious question: what happens if we train on hard puzzles?

This post documents the answer. The short version: training on hard puzzles helps slightly, but nothing we tried breaks through the fundamental ceiling. The limitation is not the training data. It is the architecture.

---

## What makes a Sudoku puzzle hard

Before running experiments, it is worth being precise about what "hard" means here.

Easy puzzles (31–36 given cells) are solvable by constraint propagation. Given the fixed cells, you can deduce the rest through two rules applied repeatedly: if a cell has only one possible value, assign it; if a unit has only one possible location for a digit, place it there. No guessing required. The solution follows logically from the givens.

Hard puzzles (17–25 given cells) cannot be solved this way. Constraint propagation gets you partway, then stalls. At that point you must guess: commit to a value for some cell, follow the consequences, and backtrack if you hit a contradiction. The hardest known puzzles require thousands of backtracks.

This is not merely a quantitative difference. It is a qualitative one. Easy puzzles require deduction. Hard puzzles require search. These are different cognitive operations, and a model trained to do one does not automatically learn the other.

All hard puzzle experiments use the `imone/sudoku-hard-v2` dataset, curated benchmarks from the competitive Sudoku solving community rated by solver backtrack count. Puzzles with rating > 50 backtracks. All evaluations use the held-out test split (44,570 puzzles) to avoid train/test overlap.

---

## Three training regimes

I tested three approaches for each architecture.

**Easy-only (baseline):** 500,000 easy puzzles, trained from scratch. This is the Post 1 model, included here for comparison.

**Finetuned on hard:** Load the easy-only checkpoint, continue training on 500,000 hard puzzles at a reduced learning rate (1e-4 instead of 1e-3). The lower rate is intended to prevent catastrophic forgetting. The model adapts to hard puzzles without overwriting what it learned about easy ones.

**Combined easy+hard:** 250,000 easy + 250,000 hard puzzles shuffled together, trained from scratch. Same total training scale as the other runs.

Both architectures (autoregressive transformer and masked diffusion) were trained under all three regimes, 20 epochs each, approximately 9 hours per run on Apple M-series.

---

## Results: masked diffusion

### Loss curves

| Model | Start loss | End loss |
|---|---|---|
| Easy-only | 0.577 | 0.021 |
| Finetuned hard | 0.359 | 0.324 |
| Combined | 0.410 | 0.213 |

The finetuned model starts at 0.359, already far better than the easy-only model's first epoch, confirming that easy puzzle pre-training transfers. But the improvement over 20 epochs is modest: 10% loss reduction compared to 97% for the easy-only run. The model is adapting slowly, not learning rapidly.

The combined model lands between the two: harder than pure easy training, much easier than pure hard.

### Hard puzzle performance

Evaluated on 1,000 held-out hard puzzles. Masked diffusion uses k=1 iterative decoding (one cell committed per pass) as the primary metric, established in Post 1 as the optimal configuration for hard puzzles.

| Model | k=1 Puzzle acc | k=1 Violations | One-shot Cell acc |
|---|---|---|---|
| Easy-only | 0.60% | 2.82 | 41.77% |
| Finetuned hard | 1.50% | 2.39 | 46.45% |
| Combined | 0.80% | 2.79 | 45.39% |

Finetuned wins across every metric. Sequential training (master easy puzzles first, then adapt to hard) outperforms simultaneous mixed training. The model that built a foundation in constraint propagation before encountering search-requiring puzzles performs better than the model that faced both simultaneously.

This result has an interesting implication for the nature of the two puzzle types. If easy and hard puzzles were simply points on a single difficulty continuum, you would expect the combined model to perform at least as well as finetuned, since it sees both difficulty levels throughout training. The fact that sequential training is better suggests the skills are genuinely different and are better learned in order.

### The ceiling

The improvements are real but modest. Finetuned achieves 1.50% puzzle accuracy versus 0.60% for easy-only, a 2.5x improvement. But this means solving 15 hard puzzles in 1000 instead of 6. The violation reduction from 2.82 to 2.39 is statistically meaningful but not qualitatively transformative.

More training does not break through. The model has approached its ceiling.

---

## Results: autoregressive transformer

### Loss curves

| Model | Start loss | End loss |
|---|---|---|
| Easy-only | [not logged] | ~0.19 |
| Finetuned hard | 0.853 | 0.823 |
| Combined | 0.636 | 0.487 |

The finetuned AR model barely moves: 0.853 to 0.823 over 20 epochs. This is qualitatively different from masked diffusion finetuning. The AR training objective asks the model to predict each token given all previous tokens. For hard puzzles where early cells are genuinely undetermined by prior context, this gradient signal is extremely noisy. The model is being asked to learn something that cannot be learned from the left-to-right perspective.

The combined AR model learns more. The easy puzzle signal provides optimization traction. But it ends at 0.487, far above the masked diffusion combined model's 0.213.

### Hard puzzle performance

| Model | Puzzle acc | Cell acc | Avg violations |
|---|---|---|---|
| AR easy-only | 0.00% | 41.72% | 25.67 |
| AR finetuned hard | 0.00% | 47.14% | 26.14 |
| AR combined | 0.00% | 45.46% | 26.35 |

The violation numbers are flat: 25.67, 26.14, 26.35. Approximately 26 violations per puzzle regardless of training regime. Cell accuracy improves with hard training (41.72% to 47.14%) but the model never learns to produce valid grids. More hard puzzle training shifts which digits the AR model predicts. It does not give the model a mechanism for global consistency.

---

## The architectural divide

Putting both architectures together:

```
Hard puzzle test set (n=1000, held-out):

                        Puzzle acc   Cell acc    Avg violations
                        (k=1 for     (one-shot)  (k=1 for
                        diffusion)               diffusion)
AR easy-only            0.00%        41.72%      25.67
AR finetuned hard       0.00%        47.14%      26.14
AR combined             0.00%        45.46%      26.35

Masked easy-only        0.60%        41.77%      2.82
Masked finetuned        1.50%        46.45%      2.39
Masked combined         0.80%        45.39%      2.79
```

The divide is definitive. No training regime brings AR violations below 25. Masked diffusion with k=1 iterative decoding stays below 3 across all training regimes. Cell accuracy is comparable between architectures (both improve with hard training to roughly 45–47%) but violations tell the opposite story.

Cell accuracy is a misleading metric for constraint satisfaction. A model can learn to predict the correct digit in more cells while still producing fundamentally invalid grids. The autoregressive model demonstrates this precisely: 47% cell accuracy with 26 violations per puzzle means it is getting nearly half of individual digits right within outputs that violate every constraint unit. Correct digits in an invalid grid are not solutions.

Violation count is the honest measure. And by that measure, the architectural difference is not a matter of degree. It is categorical.

---

## Why more data cannot fix this

The natural question: if we trained on more hard puzzles, or for more epochs, would the models eventually break through?

The answer is probably no, and the reason is structural.

Masked diffusion with iterative decoding is a greedy forward process. It commits to cells in order of confidence, one at a time, with no ability to revise committed decisions. When it makes a wrong early commitment on a hard puzzle, subsequent predictions propagate from that error. The model cannot backtrack.

This is the same fundamental limitation as Norvig's constraint propagation phase before search kicks in. Propagation alone solves easy puzzles. It stalls on hard ones. The difference is that Norvig's solver has backtracking search as a fallback. The diffusion model does not.

More data teaches the model to make better initial guesses. It does not give the model a mechanism to recover from wrong ones. The ceiling is not a data ceiling. It is an architectural one.

The autoregressive model has an even more fundamental limitation: it cannot enforce global consistency at all, regardless of how it is trained. Its outputs on hard puzzles have 26 constraint violations because it generates left to right and commits to each cell before seeing the rest of the board. No training regime can give it global information during generation.

---

## What the numbers say about hard Sudoku

Stepping back: every model we trained, regardless of architecture or training regime, fails to solve more than 1.5% of hard puzzles. The best model (masked diffusion, finetuned, k=1) solves 15 in 1000. This is not a training failure or an implementation bug. It is telling us something about the problem.

Hard Sudoku is NP-complete. What we have been building are approximate constraint propagators: models that learn to fill in cells consistently given partial information. This is a real skill, and it generalizes better from diffusion than from autoregressive generation. But approximate constraint propagation is not the same as search.

The models that succeed on easy puzzles have learned, in effect, to simulate the first phase of Norvig's solver. They can propagate constraints. They cannot search. And search is what hard puzzles require.

This is not a criticism of the approach. It is a clarification of what the approach is and what it needs to become.

---

## What comes next

The results from Post 1 showed that architecture matters for constraint learning. The results here show that training regime matters less than architecture. No amount of hard puzzle training gives a model backtracking.

The obvious question: is there an architecture that can learn to search?

Uniform diffusion is a candidate. Rather than filling in blank cells, a uniform diffusion model is trained to detect and correct errors. It receives a sequence with randomly corrupted cells and learns to identify which ones are wrong and what they should be. This is closer in spirit to iterative refinement and backtracking: look at the whole board, decide what needs fixing, fix it.

In Post 1, the uniform diffusion model failed. But that failure had a specific cause: the architecture conflated two distinct tasks (detecting which tokens are wrong and predicting what they should be) into a single output head. The gradient signal was noisy and learning stalled.

Von Rütte et al. (2025) propose a fix: explicitly separate these into two distinct predictions, one for the holding distribution (how confident is this token correct) and one for the jump chain (what should it be changed to). This disentanglement produces cleaner gradients and potentially allows the model to learn genuine iterative refinement.

That is the next experiment.

---

## Code and data

All code: [github.com/harry-david-brown](https://github.com/harry-david-brown)

Hard puzzle dataset: [HuggingFace — imone/sudoku-hard-v2](https://huggingface.co/datasets/imone/sudoku-hard-v2)

Referenced: von Rütte et al., *Scaling Behavior of Discrete Diffusion Language Models*, ICLR 2025. [arxiv.org/abs/2512.10858](https://arxiv.org/abs/2512.10858)

Lou, *Sotaku*, 2026. [github.com/chenglou/sotaku](https://github.com/chenglou/sotaku)