# Diffusion Models as Constraint Solvers: A Sudoku Benchmark

*April 2026*

---

I'm trying to use diffusion models to solve Sudoku puzzles — specifically, to find better approximate solutions within the same compute budget as classical approaches. This post documents what I built, what I found, and why the architecture matters more than I expected.

The research question: **can diffusion models learn constraint structure better than autoregressive models?**

The short answer is yes, measurably and mechanistically. Here is the evidence.

---

## Why architecture matters for constraint satisfaction

Most hard problems share a structural property: the correct value for any one component depends on the global configuration of all other components. In Sudoku, you cannot correctly fill in cell A1 without knowing the rest of the board — a digit that looks locally plausible might violate a constraint on the other side of the grid. Protein folding has the same structure. So does drug-target binding. So does circuit design.

Autoregressive models — the architecture underlying GPT, Claude, and every major LLM — generate sequences one token at a time, left to right, with no ability to revise. Each prediction is made before the rest of the sequence exists. For constraint satisfaction problems, this means making the hardest decision first with the least information.

Diffusion models work differently. They start with a noisy corrupted version of the entire output and iteratively refine it. The critical architectural difference is attention: autoregressive models use a causal mask that prevents each position from seeing future positions. Diffusion models use bidirectional attention — every position attends to every other position simultaneously. When predicting cell A1, the model has already seen cell I9.

This is the same reason GPUs beat CPUs on deep learning: the problem structure is parallel, and the hardware should match. Autoregression imposes sequential structure on problems that don't require it. Diffusion doesn't.

Diffusion models have been largely overlooked for language tasks because language modeling rewards sequential precision — reasoning, coding, instruction following — where autoregression excels. They also haven't benefited from the same scale of compute investment. But for constraint satisfaction problems, where global consistency matters more than local precision, the architectural match is different. This is the hypothesis this project tests.

---

## What I built

Three baselines, same dataset, same compute budget.

**Dataset:** Kaggle 1M Sudoku dataset (bryanpark/sudoku). 500,000 puzzles used for training. All puzzles have 31–36 given cells and are solvable by constraint propagation alone — no search required. Uniformly easy.

### Baseline 1: Classical Solver

Norvig's constraint propagation + depth-first backtracking search. Two rules applied iteratively: naked singles (if a cell has one remaining possibility, assign it) and hidden singles (if a digit can only go in one place in a unit, put it there). When propagation stalls, guess the cell with fewest candidates and recurse.

| Puzzle type | Givens | Solve time |
|---|---|---|
| Easy (n=10,000) | 31–36 | ~2ms avg, 0 requiring search |
| hard1 (Norvig) | 17 | ~30 seconds |

The classical solver is exact and fast on easy puzzles. On hard puzzles it is exact but exponentially slow.

### Baseline 2: Autoregressive Transformer

Standard transformer decoder with causal mask. Puzzle encoded as 81 integer tokens (0=blank, 1–9=digit). Model predicts all 81 solution tokens in a single left-to-right pass.

| Component | Value |
|---|---|
| Parameters | 1,071,242 |
| Embedding dim | 128 |
| Layers | 4 |
| Heads | 4 |
| Feedforward dim | 512 |
| Attention | Causal — each position attends only leftward |

Training: 500,000 puzzles, 20 epochs, Adam lr=1e-3, batch size 64. ~9 hours on Apple M-series.

### Baseline 3: Masked Diffusion

Transformer encoder with bidirectional attention. Unknown cells are randomly masked during training — the model predicts correct digits at masked positions only. Given cells are never masked. At inference, all unknown cells are masked and the model predicts them simultaneously.

| Component | Value |
|---|---|
| Parameters | 806,154 |
| Embedding dim | 128 |
| Layers | 4 |
| Heads | 4 |
| Feedforward dim | 512 |
| Attention | Bidirectional — full attention, no causal mask |
| Input vocab | 11 (digits 0–9 + MASK token) |

Training: 500,000 puzzles, 20 epochs, Adam lr=1e-3, batch size 64. ~9 hours.

### Baseline 4: Uniform Diffusion

Same encoder architecture as masked diffusion. Unknown cells are replaced with random digits rather than a mask token — the model must detect which tokens are wrong and correct them, rather than simply filling in blanks. Two tasks instead of one: error detection and error correction.

| Component | Value |
|---|---|
| Parameters | 806,026 |
| Input vocab | 10 (no MASK token needed) |
| Everything else | Identical to masked diffusion |

Training: 500,000 puzzles, 20 epochs, Adam lr=1e-3, batch size 64. ~9 hours.

---

## Results: easy puzzles

### Accuracy

| Model | Cell accuracy | Puzzle accuracy |
|---|---|---|
| AR Transformer | 96.31% | 11.88% |
| Masked Diffusion | 98.76% | 68.28% |
| Uniform Diffusion | 44.96% | 0.00% |

The diffusion model at 100k puzzles (41.82% puzzle accuracy) already outperforms the AR model at 500k puzzles (11.88%) — with 5x less training data and fewer parameters.

Uniform diffusion fails to solve any puzzles at this compute budget. The loss curve tells the story: masked diffusion converges from 0.67 to 0.02 over 20 epochs. Uniform diffusion converges from 1.14 to 0.80. The harder training signal — detecting wrong digits rather than filling in blanks — requires significantly more compute to produce useful learning. This is consistent with von Rütte et al. (2025), who note that uniform diffusion's advantage over masked diffusion only emerges at large scale.

### The position gradient

The most diagnostic result is the per-cell accuracy grid. For the autoregressive model:

```
85.6 | 88.9 | 90.0 | 91.4 | 92.3 | 93.2 | 93.3 | 93.9 | 94.7
91.7 | 92.9 | 93.9 | 93.6 | 94.7 | 95.2 | 95.2 | 95.8 | 96.4
94.2 | 95.2 | 96.0 | 95.7 | 96.4 | 96.9 | 96.2 | 96.9 | 97.3
92.2 | 93.3 | 94.1 | 95.0 | 95.7 | 96.3 | 96.6 | 97.0 | 97.4
94.8 | 95.7 | 96.5 | 96.5 | 97.3 | 97.7 | 97.7 | 98.1 | 98.4
96.5 | 97.3 | 97.8 | 97.7 | 98.2 | 98.6 | 98.3 | 98.7 | 98.9
94.9 | 95.7 | 96.2 | 97.1 | 97.7 | 98.0 | 98.4 | 98.7 | 98.9
96.6 | 97.2 | 97.8 | 98.1 | 98.5 | 98.8 | 99.0 | 99.3 | 99.4
97.8 | 98.3 | 98.8 | 99.0 | 99.2 | 99.4 | 99.4 | 99.5 | 99.6
```

A clean monotonic gradient from 85.6% (A1, top-left) to 99.6% (I9, bottom-right). The model gets better at each position simply because it has seen more of the board. This is the fingerprint of sequential commitment — not constraint reasoning, but positional pattern matching.

For the masked diffusion model:

```
98.7 | 98.8 | 98.8 | 98.9 | 98.7 | 98.7 | 98.8 | 98.8 | 98.7
98.7 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.8 | 98.8
98.7 | 98.7 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8
98.7 | 98.8 | 98.7 | 98.8 | 98.7 | 98.8 | 98.8 | 98.9 | 98.8
98.7 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.8 | 98.8
98.8 | 98.8 | 98.8 | 98.9 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8
98.8 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.7 | 98.7
98.7 | 98.7 | 98.8 | 98.8 | 98.7 | 98.8 | 98.8 | 98.7 | 98.8
98.7 | 98.7 | 98.8 | 98.7 | 98.7 | 98.7 | 98.8 | 98.7 | 98.7
```

Flat. 98.7%–98.9% across all 81 positions, a range of 0.2 percentage points. The gradient is gone because every cell has the same global context. This is the architectural difference made visible.

### Constraint violations

Cell accuracy is a misleading metric for constraint satisfaction. A model can get many individual digits correct while producing a globally incoherent output. The constraint violation rate — how often a completed grid violates a Sudoku rule — is more meaningful.

| Model | Puzzles solved | Puzzles with violations | Avg violations/puzzle |
|---|---|---|---|
| AR Transformer | 119/1000 | 881/1000 | 6.16 |
| Masked Diffusion (one-shot) | 685/1000 | 315/1000 | 1.26 |
| Masked Diffusion (k=5 iterative) | 1000/1000 | 0/1000 | 0.00 |

The autoregressive model produces 6.16 constraint violations per puzzle on average — even within its training distribution. Despite 96.3% cell accuracy, 88% of its solutions are invalid. The 11.9% puzzle accuracy comes entirely from cases where positional pattern matching happens to produce a valid grid by coincidence.

The diffusion model reduces violations by 5x in one-shot inference. With iterative decoding, violations reach exactly zero across 1000 puzzles.

### Iterative confidence-based decoding

Rather than unmasking all positions at once, iterative decoding unmasks only the k most confident predictions per pass, then reruns the model with those positions now visible. High-confidence predictions constrain uncertain ones in subsequent passes — the learned analog of Norvig's minimum remaining values heuristic.

| k | Puzzle accuracy | Avg passes | Avg violations |
|---|---|---|---|
| 1 | 100.00% | 47.2 | 0.00 |
| 5 | 100.00% | 10.0 | 0.00 |
| 10 | 99.90% | 5.0 | 0.00 |
| 20 | 99.90% | 3.0 | 0.00 |
| 81 (one-shot) | 67.40% | 1.0 | 1.26 |

k=5 is the practical optimum: 100% accuracy in 10 passes, approximately 2x slower than one-shot. The model has learned genuine confidence calibration — it knows which cells it is certain about and which are uncertain, and the iterative process exploits this to propagate constraints until the board is consistent.

---

## Results: hard puzzles

The easy dataset is uniformly easy — all puzzles solvable by constraint propagation, none requiring search. Hard Sudoku puzzles (17–25 given cells) require genuine hypothesis generation and backtracking. This is categorically different from constraint propagation, not merely harder. A model trained on easy puzzles has learned deduction. Hard puzzles require search.

We evaluated all models on 1,000 hard puzzles from `imone/sudoku-hard-v2` — curated benchmarks from the competitive Sudoku solving community, rated by solver backtrack count. Rating > 50 backtracks required.

### Out-of-distribution results

| Model | Cell accuracy | Puzzle accuracy | Avg violations |
|---|---|---|---|
| AR Transformer (one-shot) | 42.14% | 0.00% | 25.67 |
| Masked Diffusion (one-shot) | 41.89% | 0.00% | 26.22 |
| Masked Diffusion (k=1 iter) | ~37% | 0.60% | 2.77 |
| Uniform Diffusion (one-shot) | 25.15% | 0.00% | 26.52 |

In one-shot inference, AR and masked diffusion perform identically on hard puzzles. Both collapse to ~42% cell accuracy with 26+ violations per puzzle. The architectural advantage disappears when the model has no relevant training distribution to draw from.

The divergence appears with iterative decoding. k=1 iterative decoding — one cell committed per pass — reduces masked diffusion's violations from 26.22 to 2.77. It solved 6 of 1000 hard puzzles correctly. The autoregressive model has no iterative analog and solved zero.

The full k sweep:

| k | Puzzle accuracy | Avg violations |
|---|---|---|
| 1 | 0.60% | 2.77 |
| 5 | 0.30% | 5.18 |
| 10 | 0.10% | 8.19 |
| 20 | 0.00% | 14.74 |
| 81 (one-shot) | 0.00% | 26.22 |

On easy puzzles k=5 was optimal. On hard puzzles k=1 is optimal — fewer violations and higher accuracy. With only 17–25 givens, each wrong commitment cascades more aggressively. Slower unmasking gives each new token maximum context before committing.

The 2.77 violation result is the most significant finding for hard puzzles. It means most outputs are near-valid Sudoku grids — structurally close to correct even when numerically wrong. The autoregressive model's 25.67 violations are distributed across the entire grid incoherently. These are qualitatively different failure modes. One model is approximately right and locally broken. The other is globally wrong.

The position gradient also disappears on hard puzzles for the AR model — 36%–46% with no directional pattern, compared to the clean 85.6%→99.6% on easy puzzles. The memorized positional statistics don't apply, so the sequential commitment advantage vanishes. Every position is equally lost.

---

## Summary

| | AR | Masked Diffusion | Uniform Diffusion |
|---|---|---|---|
| Parameters | 1.07M | 806k | 806k |
| Easy puzzle accuracy | 11.88% | 68.28% (one-shot) / 100% (k=5) | 0.00% |
| Easy violations/puzzle | 6.16 | 1.26 (one-shot) / 0.00 (k=5) | 25.14 |
| Position gradient | 14pt range | 0.2pt range | flat ~44% |
| Hard puzzle accuracy | 0.00% | 0.60% (k=1) | 0.00% |
| Hard violations/puzzle | 25.67 | 2.77 (k=1) | 26.52 |

The research question was whether diffusion models learn constraint structure better than autoregressive models. The answer is yes.

The position gradient is the mechanism made visible: autoregressive models learn positional statistics, diffusion models learn global structure. The violation analysis confirms it: the autoregressive model's 88% violation rate on easy puzzles is not a hard puzzle problem, it is a fundamental property of sequential commitment. The diffusion model's zero violation rate with iterative decoding shows that the learned constraint structure is genuine and usable.

What we cannot yet say is whether this generalizes — whether models trained on hard puzzles can actually solve them, whether the advantage scales to problems where the constraint structure is not given explicitly. That is the question for the next post.

---

## Code and data

All code: [github.com/harry-david-brown](https://github.com/harry-david-brown)

Easy dataset: [Kaggle — bryanpark/sudoku](https://www.kaggle.com/datasets/bryanpark/sudoku)

Hard dataset: [HuggingFace — imone/sudoku-hard-v2](https://huggingface.co/datasets/imone/sudoku-hard-v2)

Referenced: von Rütte et al., *Scaling Behavior of Discrete Diffusion Language Models*, ICLR 2025. [arxiv.org/abs/2512.10858](https://arxiv.org/abs/2512.10858)
