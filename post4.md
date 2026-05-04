# On Coupling, Phase Transitions, and the Uniform Diffusion Problem

---

The six failures documented in the previous post form a pattern. Every time training became difficult, the instinct was to decouple. Separate the two heads, give each a cleaner signal, reduce interference. The endpoint of that progression is two independent models with a pipeline between them: a corruption detector followed by a masked diffusion model. Which is not more powerful than what we already have. It is masked diffusion with extra steps and extra failure modes.

But the original motivation for uniform diffusion was precisely the coupling. A model that must simultaneously reason about which tokens are wrong and what they should be might learn something that a model which only fills in blanks cannot. The coupling is not a bug to engineer around. It is the point.

This post is about why I still believe that, what the failures actually revealed, and what the right experiment looks like.

---

## The Kuramoto model

In 1975, Yoshiki Kuramoto described a system of coupled oscillators: pendulums, neurons, fireflies, any system where individual elements have their own natural frequency and are weakly connected to each other. Each oscillator would prefer to run at its own rate. The coupling pulls them toward each other. The question is what happens as you increase the coupling strength.

Below a critical coupling strength $K_c$, the oscillators run incoherently. Each does its own thing. The coupling is too weak to overcome the spread in individual frequencies. The system looks like noise.

Above $K_c$, something sudden happens. A macroscopic fraction of the oscillators spontaneously lock together in phase. They synchronize. Not because anyone told them to. Not because the coupling forced any individual oscillator into a specific state. But because the system as a whole crossed a threshold and a new stable configuration became available.

This is a phase transition. Below $K_c$: disorder. Above $K_c$: coherence. The transition is sharp.

$$r(t) = \left|\frac{1}{N}\sum_{j=1}^{N} e^{i\theta_j(t)}\right|$$

The order parameter $r$ measures how synchronized the system is. $r = 0$ means complete incoherence. $r = 1$ means perfect synchronization. Below $K_c$, $r \approx 0$. Above $K_c$, $r$ jumps discontinuously and grows with coupling strength.

The reason this matters for our problem: the holding head and the jump chain are two oscillators. The holding head has its own natural objective: learn to detect corrupted cells. The jump chain has its own: learn to predict correct digits. They are coupled through the shared transformer backbone and the shared gradient update. The question is whether that coupling is strong enough to drive synchronization, to force them to develop a joint representation that is qualitatively different from what either head would learn alone.

In our experiments, the coupling was not strong enough. Both heads learned something. They never locked together. The system stayed below $K_c$.

---

## What the failures actually revealed

The holding head, in every configuration that trained it, learned excellent corruption detection: 96% precision, 0.77+ separation between corrupted and uncorrupted cells. This is a real result. The holding head did not just learn to flag training noise, it learned something about constraint violation, about which cells are inconsistent with their neighbours given the rules of Sudoku.

The jump chain, in every configuration, plateaued at 77% correction accuracy on corrupted cells. It learned something about constraint structure. Better than random, better than the naive model, but not enough to solve puzzles.

The critical observation: the two heads never benefited from each other. The holding head's 96% precision did not help the jump chain at all. When I tried to connect them (gating), the connection hurt. The jump chain was not using the holding head's learned structure. The holding head was not using the jump chain's learned structure. They were running incoherently, each doing its own thing.

This is the signature of a system below $K_c$. The coupling through the shared backbone existed but was insufficient. The gradient signals were not strong enough, or coherent enough, to drive the two heads toward a joint representation.

---

## Why easy puzzles might be the wrong training distribution

This is the insight that reframes the previous failures.

Easy Sudoku puzzles are solvable by constraint propagation alone. Given the fixed cells, you can deduce the rest through forced moves. Cells with only one possible value, digits that can only go in one location in a unit. No uncertainty, no search, no commitment and revision.

On easy puzzles, the holding head's job is trivial: flag cells that were replaced by random digits during training. It does not need to learn anything about search or uncertainty. The training noise is the only signal it receives about which cells are wrong.

On easy puzzles, the jump chain's job is also relatively simple: given dense context (33+ given cells), predict the forced value. The constraint structure is strong enough that most corrections are overdetermined.

Neither head needs the other. The holding head can do its job without any information from the jump chain. The jump chain can do its job without any information from the holding head. The coupling is unnecessary on easy puzzles, so it never develops.

Hard Sudoku puzzles are different. They require search. At some point in solving a hard puzzle, constraint propagation stalls. There are no forced moves and the solver must commit to a hypothesis and explore its consequences. If the hypothesis is wrong, it must backtrack.

On hard puzzles, the question "which cells are wrong" is genuinely hard. Not because of training noise, but because the puzzle itself is ambiguous at the point where search begins. A cell might look locally consistent with all its neighbours but still be part of an incorrect hypothesis. The holding head, trained on hard puzzles, would need to learn something deeper than "flag random corruption". It would need to learn "flag cells that are likely part of a wrong branch of the search tree."

And the jump chain's job on hard puzzles is also harder. With only 17-25 given cells, the constraint structure is weak. The correct value for a cell is not forced by simple propagation. The jump chain needs the holding head's uncertainty estimate, needs to know which cells are confident and which are tentative, to reason correctly about what a cell should be.

On hard puzzles, the two heads need each other. The coupling becomes necessary. And when something becomes necessary for the task, the gradient forces the model to learn it.

---

## The phase transition hypothesis

The claim: there exists a training distribution on which the coupling between detection and correction becomes load-bearing, where neither head can do its job well without information from the other. And on that distribution, joint training will drive the system above $K_c$ into a synchronized, coherent joint representation.

Easy puzzles are not that distribution. Hard puzzles might be.

The experiment: train the joint ungated model from scratch on hard puzzles only. No easy pre-training, no curriculum, no gating. Just the raw joint objective on the hardest training distribution available. Let the coupling develop under the conditions that make it necessary.

This is the opposite of the usual curriculum learning intuition, start easy, progressively increase difficulty. That intuition is correct for many tasks. But for tasks where the coupling between two subtasks is the thing being learned, starting easy might actively prevent that coupling from developing. If the easy task can be solved by each head independently, the joint representation never forms.

The Sotaku paper, which I have not yet implemented, uses hard-only training on a very different architecture and reports qualitatively better results. The reason might be exactly this: hard puzzles force the model to develop representations that easy puzzles make unnecessary.

---

## The deeper argument

There is a more fundamental version of this claim, which connects to how I framed this project at the outset.

Intelligence, at some level of abstraction, is a constraint satisfaction problem. You have a space of possible states, a set of constraints that rule most of them out, and a search process that navigates toward states where all constraints are simultaneously satisfied. The question is what kind of search process you use and how it scales.

Classical solvers use explicit search with backtracking. Masked diffusion learns an approximate propagator, it can follow forced consequences but cannot search. The promise of uniform diffusion, properly implemented, is a learned search process: a model that can reason not just about what is consistent but about what is uncertain, what might need revision, where the constraint violations are hiding.

This is qualitatively different from masked diffusion. It is not a better version of the same thing. It is a different kind of computation. And that difference only becomes visible on problems that require search, problems where propagation alone is insufficient.

I have been testing a model designed to learn search on problems that do not require search. The model learned what it could from those problems and nothing more. This isn't failure, just the wrong experiment.

---

## What we are going to do

Train the joint ungated GIDD model from scratch on hard puzzles only. Same architecture as the best attempt so far (attempt 3: BCE for the holding head, CE for the jump chain, no gating). No phase 1 on easy puzzles. Straight to hard.

The prediction: the loss curves will look different. The holding head will develop a qualitatively different detection capability, not just flagging training noise but learning something about search uncertainty. The jump chain will develop tighter coupling to the holding head's representations because the task demands it. At sufficient training volume, we do not know where, there may be a phase transition where both heads' performance jumps discontinuously.

We may not have enough compute to reach that phase transition. It might be at 5M puzzles or 50M. Don't know.

What we will know after one training run: whether the joint objective on hard puzzles produces qualitatively different behavior from the joint objective on easy puzzles. Whether the separation between the heads' learned representations is different. Whether the jump chain accuracy at convergence is different. Whether the coupling, measured by how much each head's performance degrades when the other is removed, is stronger.

These are answerable questions. The answers will tell us whether the phase transition hypothesis is worth pursuing further or whether something more fundamental needs to change.

The little softmax that could.

---

Hard puzzle dataset: [HuggingFace - imone/sudoku-hard-v2](https://huggingface.co/datasets/imone/sudoku-hard-v2)

Referenced: von Rütte et al., *GIDD: Generalized Itakura-Saito Discrete Diffusion*, 2025. [arxiv.org/abs/2503.04482](https://arxiv.org/abs/2503.04482)

Kuramoto, Y., *Chemical Oscillations, Waves, and Turbulence*, 1984.

Lou, *Sotaku*, 2026. [github.com/chenglou/sotaku](https://github.com/chenglou/sotaku)

All code: [github.com/harry-david-brown](https://github.com/harry-david-brown)