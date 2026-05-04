
#  Building GIDD: Six Ways to Fail at Uniform Diffusion

In [Post 1](/post1.html), uniform diffusion failed. The model trained on 500,000 puzzles, ran for 20 epochs, and solved exactly zero of them. The loss curve barely moved. The conclusion at the time was that uniform diffusion is harder to train and needs more compute to show its advantage over masked diffusion. That was way too generous.

The problem is architectural. Our uniform diffusion model had one output head trying to do two completely different jobs simultaneously, and the resulting gradient signal was incoherent. The fix, in theory, is to separate the two jobs into two heads: one for detection, one for correction. This is what von Rütte proposes in the GIDD paper.

In practice, getting two heads to work together turns out to be harder than it sounds. In this post I'll cover six attempts to make it work, each failing in a spectacular way.

##  The two-head architecture

A uniform diffusion model receives a sequence where some cells contain wrong digits. It has two things to figure out: which cells are wrong, and what they should be. In our original model, both questions were answered by a single softmax over 10 digits. The gradient from "learn which cells are wrong" and the gradient from "learn what the correct digit is" interfered with each other. Neither task was learned cleanly.

The GIDD (Generalized Itakura-Saito Discrete Diffusion) architecture proposes separating these into two output heads:

**The holding head:** for each cell, a scalar between 0 and 1. How likely is this cell to contain a wrong digit? Trained by the IS (Itakura-Saito) divergence term of the GIDD objective.

**The jump chain head:** for each cell, a distribution over which digit it should be. Trained by the KL divergence term of the GIDD objective.

The backbone is a standard TransformerEncoder. Same bidirectional attention as the masked diffusion model, same dimensions. Only the output layer changes.

```

Input: [batch, 81] corrupted solution, digits 1-9

Embedding: [batch, 81, 128] token + position

Transformer:[batch, 81, 128] 4 layers, 4 heads, bidirectional

Jump chain: [batch, 81, 9] distribution over original digit (1-9)

Holding: [batch, 81] scalar corruption probability per cell

```

## A concrete example

Before each failure mode, it helps to have a specific picture in mind.

Say we have a four-cell puzzle with digits 1 through 4. The correct solution is $x = [3, 1, 4, 2]$. During training we corrupt some cells randomly. Cell 2 gets replaced:

$$z = [3,\; \mathbf{4},\; 4,\; 2]$$

Cell 2 now says 4 instead of 1. The model sees z and must produce two outputs.

**From the holding head,** one score per cell — the probability it is corrupted:

$$[0.05,\; \mathbf{0.95},\; 0.05,\; 0.05]$$

High for cell 2 (corrupted), low for the rest.

**From the jump chain head,** a distribution over digits for each cell. For cell 2, given that it is corrupted, the surrounding context should force the correct value:

$$\text{jump chain for cell 2:}\quad [0.90,\; 0.04,\; 0.03,\; 0.03]$$

Digit "1" with 90% probability.

The loss functions measure how far the model's outputs are from these targets. Getting the loss function right turned out to be the first, second, and third problem.

##  Attempt 1: IS divergence, both cells

The GIDD paper specifies IS divergence for the holding head. IS divergence between scalar $a$ (true corruption probability) and $b$ (model estimate) is:

$$D_{IS}(a \| b) = \frac{a}{b}  -  \log\frac{a}{b}  -  1$$

I computed it over all unknown cells, both corrupted (target $a = 1.0$) and uncorrupted (target $a = 0$, clamped to $10^{-6}$ to avoid log(0)).

```

Epoch 1: Loss 6.7953 — Jump 1.3165 — Hold 5.4788

Epoch 10: Loss 6.1337 — Jump 0.8351 — Hold 5.2986

Epoch 20: Loss 6.1015 — Jump 0.8057 — Hold 5.2958

```

The jump chain learned normally. The holding head did not move.

The problem: IS divergence with near-zero targets is numerically catastrophic. For an uncorrupted cell with target $a = 10^{-6}$ and model output $b = 0.3$:

$$D_{IS}(10^{-6} \| 0.3)  \approx  13.9$$

Every uncorrupted cell contributed ~14 to the loss. With roughly 70% of unknown cells being uncorrupted at any noise level, the holding loss was dominated by these enormous values. The gradient from uncorrupted cells (push $b$ down) fought the gradient from corrupted cells (push $b$ up). Both signals cancelled. The holding head learned nothing. IS divergence is designed for positive quantities so zero targets break it.

##  Attempt 2: IS divergence, corrupted cells only

The fix seemed obvious: only compute IS divergence where the target is 1.0 — the corrupted cells.

```

Epoch 1: Loss 1.1730 — Jump 1.1729 — Hold 0.0001

Epoch 10: Loss 0.8057 — Jump 0.8057 — Hold 0.0000

Epoch 20: Loss 0.7887 — Jump 0.7887 — Hold 0.0000

```

Holding loss hit zero immediately and stayed there. The holding head had collapsed.

The problem: with target always $a = 1.0$ and IS divergence, the global minimum is $b = 1.0$ everywhere. The model learned to output sigmoid(very large number) for every cell. $D_{IS}(1.0 \| 1.0) = 0$. Perfect loss, zero learning.

The holding head was predicting "every cell is corrupted" and being rewarded for it, because the only cells being evaluated were corrupted ones. It had no incentive to learn that uncorrupted cells should score low.

##  Attempt 3: Binary cross-entropy, both heads learning

I gave up on IS divergence and replaced it with binary cross-entropy. BCE handles zero targets cleanly, trains on both corrupted and uncorrupted cells simultaneously, and is numerically stable. This is a practical approximation to the GIDD objective rather than the exact formulation.

```

Epoch 1: Loss 1.3890 — Jump 1.1001 — Hold 0.2890

Epoch 10: Loss 1.0022 — Jump 0.7873 — Hold 0.2149

Epoch 20: Loss 0.9841 — Jump 0.7725 — Hold 0.2116

```

Both heads learning for the first time. Evaluated on 1,000 easy puzzles:

```

k= 1 — Puzzle acc: 0.00% — Cell acc: 38.76% — Avg violations: 18.94

k= 5 — Puzzle acc: 0.00% — Cell acc: 40.25% — Avg violations: 19.66

k= 81 — Puzzle acc: 0.00% — Cell acc: 45.62% — Avg violations: 25.08

```

Still zero puzzle accuracy. But something interesting happened when I measured the two heads separately:

**Holding head:**

```

Corrupted cells — mean score: 0.8719

Uncorrupted cells — mean score: 0.0953

Separation: 0.7766

Precision (threshold 0.5): 96.00%

```

The holding head had learned genuine discrimination. It assigned corrupted cells an average score of 87% and uncorrupted cells 9.5%. When it flagged a cell as corrupted, it was right 96% of the time. The detection task was essentially solved.

**Jump chain:**

```

Accuracy on corrupted cells: 76.57%

```

The jump chain was the bottleneck. At 77% accuracy per cell, the probability of all ~14 corrupted cells in a typical puzzle being corrected correctly is approximately $0.77^{14}  \approx  2\%$. This explains zero puzzle accuracy despite strong detection.

The two-head architecture had successfully separated the tasks. One task was solved but the other wasn't.

##  Attempt 4: Freeze holding head, train jump chain only

Since the holding head had converged, I froze its weights and gave the full gradient to the jump chain.

```

Epoch 1: Jump Loss: 0.7734

Epoch 10: Jump Loss: 0.7672

Epoch 20: Jump Loss: 0.7653

```

Almost no movement. The jump chain had plateaued.

The jump chain sees a corrupted token and must infer the correct value from surrounding context. The problem is that the corrupted digit is an actively misleading signal. The jump chain has to learn to discount the current cell value and reason purely from neighbours. This is harder than masked diffusion, where MASK tokens carry no misleading digit information. The jump chain had learned as much as it could from this training signal. More epochs on the same data produced nothing.

The ceiling was not a convergence problem. It was a problem with what the model was being asked to learn.

##  Attempt 5: Apply the holding gate at inference only

The holding head can identify corrupted cells with 96% precision. What if we used those detections to replace corrupted tokens with a MASK token before the jump chain sees them, at inference time without retraining?

I extended the vocabulary to include a MASK token, loaded the checkpoint with a partial weight transfer and ran inference with the gate active.

```

k= 1 — Cell acc: 15.23% — Avg violations: 26.88

k= 5 — Cell acc: 14.47% — Avg violations: 26.94

k= 81 — Cell acc: 11.46% — Avg violations: 27.00

```

Cell accuracy dropped to approximately random (11% = 1/9 digits). The gate made things dramatically worse.

The reason is obvious in retrospect. The jump chain was trained on sequences like `[3, 4, 4, 2]` — corrupted digits, no masks. Showing it `[3, MASK, 4, 2]` at inference was completely out of distribution. It had never seen a MASK token. Its internal representations had no meaning for that embedding.

The training mismatch was the bottleneck. The jump chain must be trained with masked inputs to benefit from masking at inference.

##  Attempt 6: Train with gating from the start

Clean slate. Vocabulary extended to include MASK from the beginning. Both heads trained jointly with the holding gate active during every forward pass. The holding head flags cells, those positions get MASK embeddings, then the jump chain sees the masked sequence and predicts corrections.

Training took twice as long, which makes sense because the transformer should run twice per forward pass.

```

Epoch 1: Loss 1.9566 — Jump 1.5951 — Hold 0.3615

Epoch 10: Loss 1.4270 — Jump 1.2032 — Hold 0.2238

Epoch 20: Loss 1.4050 — Jump 1.1851 — Hold 0.2199

k= 1 — Cell acc: 27.32% — Avg violations: 23.38

k= 5 — Cell acc: 26.13% — Avg violations: 24.42

k= 81 — Cell acc: 16.21% — Avg violations: 26.81

Holding head separation: 0.7674

```

The holding head learned. The jump chain at 1.185 loss was significantly worse than ungated training (0.773). Better than random. Worse than every previous jump chain attempt.

The problem: bootstrapping. Training the holding head and the gated jump chain simultaneously creates a chicken-and-egg problem. The jump chain needs a reliable holding head to learn from masked inputs. In early epochs the holding head is noisy and flags the wrong cells. The jump chain receives inconsistent training signal — sometimes MASK at corrupted positions, sometimes MASK at uncorrupted positions, sometimes real digits at corrupted positions. The incoherent signal makes the jump chain harder to optimize than either pure noisy-digit training or pure masked training.

##  Summary

Six attempts. Six different failure modes.

The holding head works in every configuration that trains it. 96% precision, 0.77+ separation, consistent across attempts 3 through 6. So that isn't the problem.

The jump chain is the problem, and it's specific: the jump chain needs masked inputs to learn to reason from context rather than from the corrupted digit. But the mask must come from a reliable holding head, which does not exist at the start of training. Training them simultaneously creates a bootstrapping failure. Training them sequentially leaves the jump chain at 77% accuracy because switching to masked inputs after ungated convergence does not add new signal.

While our masked diffusion model solves 100% of easy puzzles in 10 passes, our GIDD implementation, after six training runs and approximately 30 hours of compute, has not solved a single one.

Back to the drawing board!

---


Referenced: von Rütte et al., *Scaling Behavior of Discrete Diffusion Language Models*, ICLR 2025. [arxiv.org/abs/2512.10858](https://arxiv.org/abs/2512.10858)

von Rütte et al., *GIDD: Generalized Itakura-Saito Discrete Diffusion*, 2025. [arxiv.org/abs/2503.04482](https://arxiv.org/abs/2503.04482)

All code: [github.com/harry-david-brown](https://github.com/harry-david-brown)