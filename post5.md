# What the Six Failures Actually Tested

---

[Post 4](/post4.html) ended on a prediction. Train the two-head model on hard puzzles only, and the detection head and the correction head might be forced to need each other. I said one training run would tell us whether that was worth pursuing.

I ran it. Then I read the GIDD paper properly, which I should have done before building anything. This post covers both, and the second one changes how I read the first.

---

## The result

Same architecture as attempt 3 in [Post 3](/post3.html): shared encoder, a holding head trained with binary cross-entropy, a jump chain head trained with cross-entropy, no gating. Trained from scratch on 500,000 hard puzzles for 20 epochs. Evaluated on 1,000 held-out hard puzzles.

```
Hard puzzle test set (k=1):

                           Puzzle acc   Cell acc   Avg violations
Masked easy-only           0.60%        35.81%     2.82
Masked finetuned           1.50%        36.82%     2.39
Two-head, hard-only        0.00%        21.23%     17.95
```

Worse than masked diffusion on every measure. Holding separation fell to 0.693 (from 0.777 on easy puzzles) and jump chain accuracy on corrupted cells fell to 57% (from 77%). No jump, no synchronization. The loss was still drifting down at epoch 20, so I can't say it had converged, but nothing in the curve looked like a transition on its way.

On its face, that is a clean negative result for the phase transition hypothesis. I don't think it is one, and the reason is embarrassing.

---

## What the paper actually says

I had been working from von Rütte's blog post and my own reading of the ELBO. When I went through the paper itself, four things did not match what I built.

**1. The paper's model has one output, not two.** A single network predicts a distribution over the clean token. Both terms of the loss, the KL term and the Itakura-Saito term, are computed from that one output. The two-head split is a conjecture from the blog post, not something the paper implements or tests. The paper doesn't use the words "holding" or "jump chain" at all.

This matters because in the one-output version the two questions cannot come apart. The model's belief that the current digit is right is one entry of the distribution. Its belief about what the digit should be is the rest. They sum to one. My two heads were two separate layers with two separate losses, and nothing in the objective tied them together. In Post 4 I wrote that the heads "never benefited from each other" and treated that as a finding about coupling. It was a property of the loss I wrote.

**2. The paper never trains on wrong digits alone.** Its noise is mostly masking with a small share of random tokens, 10 to 20 percent at the midpoint. Generation starts from all masks. I trained on pure random digits and started inference from a grid of random digits. That is the hardest corner of the design space, and the paper doesn't go there.

**3. The paper's model is told how noisy the input is.** The network takes the noise level as an input. Mine didn't. At the start of inference nearly every unknown cell is garbage, and my model had no way to know that.

**4. My sampler could not revise.** This is the one that stings. The whole argument for uniform diffusion, the argument I made at the end of [Post 2](/post2.html), is that a model trained to spot wrong tokens can go back and change them. My inference loop picked cells, overwrote them once, and locked them. It also chose which cells to commit by asking the holding head which looked most corrupted, which at the start is all of them. I built a model whose purpose was backtracking and then ran it with the same no-undo rule as masked diffusion.

Two smaller things. The loss weighting the paper spends a section on (the raw weights explode at very low and very high noise, and taming them is a large part of what it calls GIDD+) never applied to me, because my loss had no weights. And attempts 1 and 2 in Post 3 failed for a reason I misread: the paper applies Itakura-Saito divergence to a probability that is never zero, and I applied it to a 0-or-1 label.

---

## Rereading the earlier posts

**Post 1.** The single-head uniform model scored 0% and I concluded it needed more compute. In Post 3 I revised that to "one head doing two jobs." Neither holds up. One head is what the paper uses. That model was missing the same things: mixed noise, a noise-level input, and a sampler that revises.

**Post 3.** The six failures are real, and the bootstrapping problem in attempt 6 is real. But they are six ways to fail at a two-head detector-plus-corrector, which is closer to ELECTRA than to GIDD. The title should have been more modest.

**Post 4.** The hypothesis was that hard puzzles make the coupling load-bearing. I still find it plausible. But the experiment could not test it, because the model had no channel through which the coupling could form and no way to act on it at inference.

One piece of context I left out before: the paper's own results don't promise this works at my scale. Its models trained with uniform noise have consistently higher loss than mask-only ones, with a small trend toward closing the gap as models grow. Its smallest model has 28 million parameters. Mine has 800 thousand.

---

## What I can and can't say

I can say that masked diffusion beats autoregression on Sudoku, that the position gradient shows why, and that neither solves hard puzzles. Posts 1 and 2 stand.

I can't say that GIDD fails on Sudoku. I haven't run GIDD on Sudoku.

---

## What comes next

Build the thing the paper describes, in two stages.

**Stage 1: the loss.** One output head. Hybrid noise with a configurable share of random digits. Noise level as an input. The weighted two-term loss. The paper offers a built-in check: with the random share set to zero, the model should reproduce masked diffusion. So the first milestone is the new code matching the old result, 100% on easy puzzles at k=5.

**Stage 2: the sampler.** Turn the random share up to 10 or 20 percent and decode with a sampler that can change any cell, including the paper's self-correction step. Then run it against the hard test set, where masked diffusion's best is 1.5%.

If it clears that bar, the revision argument from Post 2 has its first evidence. If it doesn't, I will finally have a negative result about the right model.

Read the paper first. Then build.

---

Hard puzzle dataset: [HuggingFace - imone/sudoku-hard-v2](https://huggingface.co/datasets/imone/sudoku-hard-v2)

Referenced: von Rütte et al., *Generalized Interpolating Discrete Diffusion*, ICML 2025. [arxiv.org/abs/2503.04482](https://arxiv.org/abs/2503.04482)

von Rütte, *Why Diffusion Language Models Are the Future*, 2026. [dimitri.ml](https://dimitri.ml/posts/why-diffusion-language-models-are-the-future/)

All code: [github.com/harry-david-brown](https://github.com/harry-david-brown)
