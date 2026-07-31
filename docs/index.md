---
layout: technical
title: The Policy Triangle
short_title: The Policy Triangle
subtitle: A Taxonomy of Policy Mismatch in LLM Reinforcement Learning
description: A technical taxonomy of behavior, proximal, and current policy mismatch in LLM reinforcement learning.
author: Huaiyi Zhao
date: 2026-07-31
---

<figure class="hero-figure">
  <img src="{{ '/assets/policy-mismatch-intro.svg' | relative_url }}" alt="A response passes through three versions of an AI model: the behavior model writes it, the proximal model marks the start of learning, and the current model changes while learning. The policy triangle tracks the resulting mismatch.">
</figure>

<div class="abstract" markdown="1">

## Abstract

Large-language-model reinforcement learning increasingly separates rollout
generation from optimization, creating mismatch among the behavior policy that
generated a token, the proximal policy that anchors an update, and the current
policy being trained. Existing methods use importance weights, clipping, and
rejection, but are difficult to compare because similar ratios play different
roles. This note organizes these mechanisms with the policy triangle:
\\(B=q/\mu\\), \\(U=\pi_\theta/q\\), and \\(E=\pi_\theta/\mu=BU\\). An intervention is
described by an edge, an operator—detached mask, detached weight, or
differentiable update shaper—and its token-, sequence-, or group-level
geometry. The result is a common taxonomy that separates mismatch source from
intervention semantics across direct and decoupled methods, not a new optimizer
or convergence claim.

</div>

## 1. Motivation {#motivation}

Classical policy gradients begin with REINFORCE
([Williams, 1992](https://doi.org/10.1007/BF00992696)); TRPO and PPO add
proximal control so that sampled data can be reused without arbitrarily large
updates ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477);
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). PPO and
group-relative variants remain common in LLM training
([Shao et al., 2024](https://arxiv.org/abs/2402.03300);
[Yu et al., 2025](https://arxiv.org/abs/2503.14476)), while REINFORCE-style
objectives such as ReMax, RLOO, CISPO, and TOPR have renewed interest
([Li et al., 2023](https://arxiv.org/abs/2310.10505);
[Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740);
[MiniMax et al., 2025](https://arxiv.org/abs/2506.13585);
[Le Roux et al., 2025](https://arxiv.org/abs/2503.14286)).

At system scale, inference and training are usually separate
([Sheng et al., 2024](https://arxiv.org/abs/2409.19256);
[Fu et al., 2025](https://arxiv.org/abs/2505.24298)). A rollout may be generated
by stale weights, a different numerical backend, lower precision, or different
mixture-of-experts routing. Thus the probability recorded by the rollout engine
need not equal the probability recomputed by the learner, even when both
nominally refer to the same checkpoint
([Qiu et al., 2026](https://arxiv.org/abs/2601.18150);
[Zhong et al., 2026](https://arxiv.org/abs/2605.14220)). Several optimizer
updates can further move the trainable policy before the sample is consumed
([Zhang et al., 2026](https://arxiv.org/abs/2602.01826)).

Recent methods address these effects with a rapidly expanding vocabulary:
decoupled PPO ([Hilton et al., 2022](https://arxiv.org/abs/2110.00641)), TIS and
MIS ([Ionides, 2008](https://doi.org/10.1198/106186008X320456);
[veRL contributors, 2026](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)),
IcePop and KPop ([Ling Team et al., 2025](https://arxiv.org/abs/2510.18855);
[Guo et al., 2026](https://ringtech.notion.site/kpop)), GSPO
([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)), DPPO
([Qi et al., 2026](https://arxiv.org/abs/2602.04879)), TRM
([Li et al., 2025](https://arxiv.org/abs/2512.23075)), ESTR
([Zhao et al., 2026](https://arxiv.org/abs/2607.22186)), and A-3PO
([Li et al., 2025](https://arxiv.org/abs/2512.06547)). The mechanisms are often
presented as unrelated losses, obscuring whether an operation selects data,
changes measure, or constrains the current update.

> **Position of this note.** The contribution is a coordinate system for
> comparing existing mechanisms. It is deliberately narrower than a full
> theory of off-policy RL and does not claim a new optimizer, performance
> result, or convergence theorem.

## 2. The policy triangle {#triangle}

For prompt \\(x\\), response \\(y=(y_1,\ldots,y_T)\\), and token context
\\(h_t=(x,y_{\lt t})\\), define three conditional policies:

<div class="equation">
\[
\mu_t=\mu(y_t\mid h_t),\qquad
q_t=q(y_t\mid h_t),\qquad
\pi_t=\pi_\theta(y_t\mid h_t).
\]
</div>

The corresponding token ratios are

<div class="equation">
\[
B_t=\frac{q_t}{\mu_t},\qquad
U_t=\frac{\pi_t}{q_t},\qquad
E_t=\frac{\pi_t}{\mu_t}=B_tU_t.
\tag{1}
\]
</div>

| Edge | Ratio | Causal interpretation |
|---|---|---|
| Behavior edge | \\(B=q/\mu\\) | Mismatch already present at learner admission: staleness and train–inference disagreement. |
| Update edge | \\(U=\pi/q\\) | Drift created by the current optimization step relative to its proximal anchor. |
| Direct edge | \\(E=\pi/\mu=BU\\) | Total behavior-to-current mismatch. |

<figure class="triangle-figure">
  <img src="{{ '/assets/policy-triangle-hero.svg' | relative_url }}" alt="Graphical abstract of the policy triangle: behavior policy mu, proximal policy q, and current policy pi connected by B, U, and E equals B times U, alongside mask, weight, and clip operators.">
  <figcaption>
    Figure 1. Three policy anchors, their edge ratios, and the operators that
    can act on them. Each attachment independently chooses an edge, an operator,
    and a token-, sequence-, or group-level geometry.
  </figcaption>
</figure>

### 2.1 Bypass and factorization

**Bypass.** Set \\(q\equiv\mu\\). Then \\(B=1\\) and \\(U=E\\). Statistical correction
and proximal control are compressed onto one direct edge. This avoids a
proximal forward pass but loses causal attribution.

**Decoupled.** Keep a distinct frozen \\(q\\). The learner can correct rollout
mismatch on \\(B\\) while constraining the current update on \\(U\\), typically as
\\(\mathrm W(B)\mathrm C(U)\\).

The proximal anchor is itself a design variable. A-3PO, for example,
approximates it by interpolating behavior and current log-probabilities to avoid
an additional model forward pass while retaining the factorization \\(E=BU\\)
([Li et al., 2025](https://arxiv.org/abs/2512.06547)).

## 3. Edge operators and geometry {#operators}

For any edge \\(X\in\{B,U,E\}\\), distinguish operations by their gradient
semantics rather than by their names in a particular implementation.

<div class="table-wrap" markdown="1">

| Operator | Definition | Gradient semantics | Typical role |
|---|---|---|---|
| Mask \\(\mathrm M\\) | \\(\mathrm M_{g,s}(X,\hat A)\in\{0,1\}\\) | Detached | Select a token, sequence, or group using a ratio band, K2/K3 statistic, divergence, age, or metadata. |
| Weight \\(\mathrm W\\) | \\(\mathrm W_{g,s}(X)=\operatorname{sg}[f_{g,s}(X)]\ge0\\) | Detached | Change measure using raw, truncated, tapered, or self-normalized importance sampling. |
| Shaper \\(\mathrm C\\) | Enters the loss through \\(\pi_\theta\\) | Differentiable | Constrain or reshape the current update using clipping, smooth gates, or divergence geometry. |

</div>

PPO's effective token coefficient is advantage-sign dependent:

<div class="equation">
\[
\mathrm C^{\mathrm{PPO}}(X,\hat A)=
X\,\mathbf 1\!\left[
(\hat A\ge0 \land X\le1+\epsilon_+)
\;\lor\;
(\hat A&lt;0 \land X\ge1-\epsilon_-)
\right].
\tag{2}
\]
</div>

Selecting the clipped branch removes gradient in a particular direction. This
differs from truncating a detached importance weight. DAPO changes the
sign-dependent bounds ([Yu et al., 2025](https://arxiv.org/abs/2503.14476));
GSPO replaces the token ratio with a length-normalized sequence-geometric ratio
([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)).

### 3.1 Geometry is local to an attachment

An algorithm need not have one global granularity. Each operator may
independently use:

- **Token ratio:** \\(X_t\\).
- **Sequence product:** \\(R_X=\prod_tX_t\\), an exact trajectory ratio but
  usually high variance.
- **Geometric ratio:** \\(G_X=\exp(T^{-1}\sum_t\log X_t)\\), length normalized but
  not a trajectory density ratio.
- **Group statistic:** aggregation over several responses from one prompt.
- **Divergence:** sampled binary KL, top-\\(k\\), or full-distribution KL/TV.

A broad class of critic-free policy-loss gradients can be written as

<div class="equation">
\[
\widehat g=
\frac{1}{Z}\sum_i
\mathrm M_i\,\overline{\mathrm W}_i\,\mathrm C_i\,
\hat A_i\nabla_\theta\log\pi_\theta(y_i\mid h_i).
\tag{3}
\]
</div>

A concrete attachment is specified by
\\((\text{edge},\text{operator},\text{scope},\text{statistic},
\text{sign rule},\text{normalization})\\).

## 4. Composing interventions {#composition}

The factorization \\(E=BU\\) does not make interventions interchangeable. Where
an operator is attached determines what mismatch it sees, how often it must be
recomputed, and which bias–variance trade-off it introduces.

### 4.1 Nonlinear placement matters

<div class="equation">
\[
\operatorname{clip}(E)\ne
\operatorname{clip}(B)\operatorname{clip}(U),\qquad
\mathrm M(E)\ne\mathrm M(B)\mathrm M(U).
\tag{4}
\]
</div>

For \\(k_3(x)=x-1-\log x\\),

<div class="equation">
\[
k_3(BU)=k_3(B)+k_3(U)+(B-1)(U-1).
\tag{5}
\]
</div>

Therefore, filtering total mismatch is not equivalent to filtering the behavior
and update edges independently. A mask may nevertheless read any edge without
forcing a downstream importance weight to consume the same edge.

### 4.2 Variance control

Truncated importance sampling uses \\(\widetilde w_i=\min(w_i,c)\\)
([Ionides, 2008](https://doi.org/10.1198/106186008X320456)). Self-normalization
is orthogonal:

<div class="equation">
\[
\overline w_i=
\frac{\widetilde w_i}
{\sum_j M_j\widetilde w_j/\sum_jM_j}.
\tag{6}
\]
</div>

Its domain—token, sequence, prompt group, or batch—is part of the algorithm. It
stabilizes mean gradient scale but is biased at finite sample size
([Owen, 2013](https://artowen.su.domains/mc/)).

### 4.3 A bookkeeping sanity check

When an estimator is specifically intended to reproduce the direct
behavior-to-current change-of-measure kernel, its unshaped limit should reduce
to \\(E=BU\\). This conditional check can catch accidental duplication: for
example, \\(\mathrm W(E)\mathrm C(E)\\) reduces to \\(E^2\\), whereas the factorized
construction \\(\mathrm W(B)\mathrm C(U)\\) reduces to \\(BU\\).

This is an implementation diagnostic, not a correctness or unbiasedness
criterion. Clipping, masking, geometric aggregation, and self-normalization may
deliberately move an estimator away from that limit.

## 5. Existing methods on the triangle {#taxonomy}

The table isolates policy mismatch. It does not enumerate reward shaping,
advantage construction, entropy bonuses, dynamic sampling, or systems
optimizations.

<div class="table-wrap" markdown="1">

| Method family | Topology | Triangle placement | Interpretation |
|---|---|---|---|
| [PPO](https://arxiv.org/abs/1707.06347) / [GRPO](https://arxiv.org/abs/2402.03300) / [DAPO](https://arxiv.org/abs/2503.14476) | Coupled | \\(\mathrm C_{\mathrm{tok}}(E)\\) | One direct ratio carries correction and proximal-control roles; DAPO uses asymmetric bounds. |
| [CISPO](https://arxiv.org/abs/2506.13585) / [TOPR](https://arxiv.org/abs/2503.14286) | Direct PG | \\(\mathrm W_{\mathrm{tok}}(E)\\) | Detached clipped or tapered weight; gradients need not vanish outside a PPO band. |
| [GSPO](https://arxiv.org/abs/2507.18071) | Coupled | \\(\mathrm C_{\mathrm{geo}}(E)\\) | Sequence-geometric shaping; the original objective does not add a separate pre-filter. |
| [Decoupled PPO](https://arxiv.org/abs/2110.00641) / [AReaL](https://arxiv.org/abs/2505.24298) / [A-3PO](https://arxiv.org/abs/2512.06547) | Factorized | \\(\mathrm W(B)\mathrm C(U)\\) | Behavior correction is detached from the trainable proximal constraint. |
| [TIS](https://doi.org/10.1198/106186008X320456) / MIS / [rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html) | Either | \\(\mathrm M(B\text{ or }E)\mathrm W(B\text{ or }E)\\), optional \\(\mathrm C\\) | Weighting and rejection compose; bypass and decoupled modes change edge semantics. |
| [IcePop](https://arxiv.org/abs/2510.18855) / [KPop](https://ringtech.notion.site/kpop) | Usually direct | \\(\mathrm M_{\mathrm{tok}}(E)\\) plus a base loss | IcePop uses a ratio region; KPop uses bidirectional binary-KL geometry sensitive to absolute probability. |
| [DPPO](https://arxiv.org/abs/2602.04879) / [TRM](https://arxiv.org/abs/2512.23075) | Direct | Divergence \\(\mathrm M(E)\\) or \\(\mathrm C(E)\\) | Distributional geometry replaces sampled-ratio magnitude; TRM rejects a sequence on worst-token divergence. |

</div>

### 5.1 Three recurring ambiguities

1. **Bypass is not “IS disabled.”** It means \\(B=1\\); the direct loss may still
   contain \\(E=U\\).
2. **PPO and GSPO clipping are not pre-filters.** Their branches change gradient
   flow in an advantage-dependent direction; a detached mask rejects
   independently of that direction.
3. **A decoupled filter may read any raw edge.** \\(\mathrm M(B)\\) is static when
   \\(q\\) is frozen, whereas \\(\mathrm M(U)\\) and \\(\mathrm M(E)\\) generally
   change as \\(\pi\\) is updated.

### 5.2 Relation to nearby views

[Decoupled PPO](https://arxiv.org/abs/2110.00641) is the closest precursor
because it explicitly inserts a proximal policy between behavior and current
policies. [AReaL](https://arxiv.org/abs/2505.24298) and
[A-3PO](https://arxiv.org/abs/2512.06547) operationalize or approximate that
factorization for asynchronous LLM training. The triangle is broader in
operator placement but does not prescribe a particular proximal anchor.

The [veRL rollout-correction formulation](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
and [FP8-RL](https://arxiv.org/abs/2601.18150) compare practical TIS, MIS,
rejection, bypass, and decoupled configurations. The triangle abstracts that
implementation pipeline and additionally places direct PPO, GSPO, and
divergence-based shapers in the same coordinates.

[RPG](https://arxiv.org/abs/2505.17508) and the
[off-policy interpretation of group-relative REINFORCE](https://arxiv.org/abs/2509.24203)
give deeper analyses of KL regularization, baselines, and data shaping. Those
axes are intentionally outside this critic-free policy-loss taxonomy.
[Jackpot](https://arxiv.org/abs/2602.06107) also lies partly outside the view
because it changes the behavior distribution through rejection and joint
rollout-model updates, rather than only operating on stored trajectories
downstream.

## 6. Design implications and hypotheses {#design}

A useful heuristic is to place detached statistical correction near the
behavior anchor and differentiable control near the current anchor. Mismatch on
\\(B\\) has already happened and can only be selected or reweighted. Drift on
\\(U\\) is produced by the current optimizer and is therefore a natural target
for a trainable trust region.

### 6.1 Partial and piecewise behavior policies

A resumed or interrupted trajectory may be generated by several policy
versions. Let \\(v(t)\\) denote the version used at token \\(t\\):

<div class="equation">
\[
\mu(y\mid x)=\prod_t\mu_{v(t)}(y_t\mid h_t),\qquad
B_t=\frac{q_t}{\mu_{v(t),t}}.
\tag{7}
\]
</div>

Stored per-token sampling log-probabilities support the observed conditional
ratio, but do not correct unlogged prompt selection, replay selection, or
resume-selection bias.

### 6.2 Taxonomy-derived research hypotheses

> **Novelty and evidence caveat.** To the best of the literature search through
> July 2026, the exact compositions below were not found as named, evaluated
> methods in the cited work. They are design templates, not priority or
> effectiveness claims.

1. **Decoupled sequence shaping:**
   \\(\mathrm W_{\mathrm{tok}}(B)\mathrm C_{\mathrm{geo}}(U)\\). Correct rollout
   mismatch token-wise, then constrain the update with a sequence-geometric
   proximal ratio.
2. **Heterogeneous two-edge control:** entropy-adaptive admission and
   \\(\mathrm W(B)\\) on the behavior edge, followed by a divergence mask and
   ordinary \\(\mathrm C(U)\\) on the update edge.
3. **Segment-aware asynchronous correction:** normalize or truncate
   \\(\mathrm W(B_t)\\) within policy-version segments, optionally gate the whole
   sequence using \\(E\\), and retain \\(\mathrm C(U)\\) for the current update.

### 6.3 Implementation audit

1. **Identify the actual behavior distribution.** Include sampling transforms,
   backend precision, and piecewise versions.
2. **Identify the proximal anchor.** State whether it equals behavior, is a
   frozen snapshot, or is approximated.
3. **Record every attachment.** Edge, detachment, statistic, scope, sign rule,
   and normalization domain.
4. **Inspect composed ratios.** When implementing behavior-to-current
   correction, check the unshaped kernel for missing or duplicated edge factors.
5. **Instrument raw ratios before truncation.** Report \\(B\\), \\(U\\), \\(E\\),
   rejection, effective sample size, gradient variance, age, reward, and
   throughput.

## 7. Scope and conclusion {#scope}

The triangle does not choose thresholds, guarantee that masking improves
return, repair support mismatch, or replace empirical validation. It omits
value/advantage off-policy correction, reward-model drift, environment
nonstationarity, and optimizer-state staleness. Sequence-product importance
sampling can be formally exact and statistically unusable; geometric statistics
can be stable while not being exact change-of-measure factors.

Within this scope, the behavior, proximal, and current policies provide a
compact set of anchors. Masks select regions, detached weights change measure,
and differentiable shapers control the update; each chooses its own geometry.
Bypass collapses the triangle, while decoupling separates exogenous rollout
mismatch from endogenous update drift.

The main value is comparative. The edge–operator–geometry representation keeps
mismatch source, intervention semantics, and statistical scale distinct. It
places published mechanisms in a common coordinate system and makes
underexplored combinations easier to state without presenting them as
established improvements.

### Suggested citation {#citation}

    @misc{zhao2026policytriangle,
      title  = {The Policy Triangle: A Taxonomy of Policy Mismatch
                in LLM Reinforcement Learning},
      author = {Huaiyi Zhao},
      year   = {2026},
      url    = {https://huaiyizhao.github.io/policy-triangle/}
    }

## References {#references}

<div class="references-list" markdown="1">

1. Ahmadian, A. et al. “Back to Basics: Revisiting REINFORCE Style Optimization
   for Learning from Human Feedback in LLMs.” *arXiv*, 2024.
   [2402.14740](https://arxiv.org/abs/2402.14740).
2. Chen, Z. et al. “Jackpot: Optimal Budgeted Rejection Sampling for Extreme
   Actor–Policy Mismatch Reinforcement Learning.” *ICLR 2026*.
   [2602.06107](https://arxiv.org/abs/2602.06107).
3. DeepSeek-AI et al. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
   via Reinforcement Learning.” *arXiv*, 2025.
   [2501.12948](https://arxiv.org/abs/2501.12948).
4. Fu, W. et al. “AReaL: A Large-Scale Asynchronous Reinforcement Learning
   System for Language Reasoning.” *arXiv*, 2025.
   [2505.24298](https://arxiv.org/abs/2505.24298).
5. Guo, J. et al. “KPop: Taming Training–Inference Mismatch in Reinforcement
   Learning with Adaptive Masking Regions.” Technical report, 2026.
   [Project page](https://ringtech.notion.site/kpop).
6. Hilton, J., Cobbe, K., and Schulman, J. “Batch Size-Invariance for Policy
   Optimization.” *NeurIPS*, 2022.
   [2110.00641](https://arxiv.org/abs/2110.00641).
7. Ionides, E. L. “Truncated Importance Sampling.” *Journal of Computational
   and Graphical Statistics*, 2008.
   [doi](https://doi.org/10.1198/106186008X320456).
8. Le Roux, N. et al. “Tapered Off-Policy REINFORCE: Stable and Efficient
   Reinforcement Learning for LLMs.” *arXiv*, 2025.
   [2503.14286](https://arxiv.org/abs/2503.14286).
9. Li, X., Wu, S., and Shen, Z. “A-3PO: Accelerating Asynchronous LLM Training
   with Staleness-Aware Proximal Policy Approximation.” *arXiv*, 2025.
   [2512.06547](https://arxiv.org/abs/2512.06547).
10. Li, Y. et al. “Trust Region Masking for Long-Horizon LLM Reinforcement
    Learning.” *arXiv*, 2025.
    [2512.23075](https://arxiv.org/abs/2512.23075).
11. Li, Z. et al. “ReMax: A Simple, Effective, and Efficient Reinforcement
    Learning Method for Aligning Large Language Models.” *arXiv*, 2023.
    [2310.10505](https://arxiv.org/abs/2310.10505).
12. Ling Team et al. “Every Step Evolves: Scaling Reinforcement Learning for
    Trillion-Scale Thinking Model.” *arXiv*, 2025.
    [2510.18855](https://arxiv.org/abs/2510.18855).
13. MiniMax et al. “MiniMax-M1: Scaling Test-Time Compute Efficiently with
    Lightning Attention.” *arXiv*, 2025.
    [2506.13585](https://arxiv.org/abs/2506.13585).
14. Ouyang, L. et al. “Training Language Models to Follow Instructions with
    Human Feedback.” *NeurIPS*, 2022.
    [2203.02155](https://arxiv.org/abs/2203.02155).
15. Owen, A. B. *Monte Carlo Theory, Methods and Examples*, 2013.
    [Online book](https://artowen.su.domains/mc/).
16. Qi, P. et al. “Rethinking the Trust Region in LLM Reinforcement Learning.”
    *arXiv*, 2026. [2602.04879](https://arxiv.org/abs/2602.04879).
17. Qiu, Z. et al. “FP8-RL: A Practical and Stable Low-Precision Stack for LLM
    Reinforcement Learning.” *arXiv*, 2026.
    [2601.18150](https://arxiv.org/abs/2601.18150).
18. Schulman, J. et al. “Trust Region Policy Optimization.” *ICML*, 2015.
    [1502.05477](https://arxiv.org/abs/1502.05477).
19. Schulman, J. et al. “Proximal Policy Optimization Algorithms.” *arXiv*,
    2017. [1707.06347](https://arxiv.org/abs/1707.06347).
20. Shao, Z. et al. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning
    in Open Language Models.” *arXiv*, 2024.
    [2402.03300](https://arxiv.org/abs/2402.03300).
21. Sheng, G. et al. “HybridFlow: A Flexible and Efficient RLHF Framework.”
    *arXiv*, 2024. [2409.19256](https://arxiv.org/abs/2409.19256).
22. veRL contributors. “Rollout Correction: Mathematical Formulations and
    Configuration Guide.” 2026.
    [Documentation](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html).
23. Williams, R. J. “Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning.” *Machine Learning*, 1992.
    [doi](https://doi.org/10.1007/BF00992696).
24. Yao, C. et al. “Group-Relative REINFORCE Is Secretly an Off-Policy
    Algorithm.” *ICLR 2026*.
    [2509.24203](https://arxiv.org/abs/2509.24203).
25. Yu, Q. et al. “DAPO: An Open-Source LLM Reinforcement Learning System at
    Scale.” *arXiv*, 2025.
    [2503.14476](https://arxiv.org/abs/2503.14476).
26. Zhang, Y. et al. “Beyond Precision: Training–Inference Mismatch Is an
    Optimization Problem and Simple LR Scheduling Fixes It.” *arXiv*, 2026.
    [2602.01826](https://arxiv.org/abs/2602.01826).
27. Zhang, Y. et al. “On the Design of KL-Regularized Policy Gradient
    Algorithms for LLM Reasoning.” *ICLR 2026*.
    [2505.17508](https://arxiv.org/abs/2505.17508).
28. Zhao, G. et al. “Deconstructing Off-Policy Ratios: Entropy-Scaled Trust
    Regions for Asynchronous Reinforcement Learning.” *arXiv*, 2026.
    [2607.22186](https://arxiv.org/abs/2607.22186).
29. Zhao, S. et al. “Large Language Model Post-Training: A Unified View of
    Off-Policy and On-Policy Learning.” *arXiv*, 2026.
    [2604.07941](https://arxiv.org/abs/2604.07941).
30. Zheng, C. et al. “Group Sequence Policy Optimization.” *arXiv*, 2025.
    [2507.18071](https://arxiv.org/abs/2507.18071).
31. Zhong, T. et al. “Diagnosing Training Inference Mismatch in LLM
    Reinforcement Learning.” *arXiv*, 2026.
    [2605.14220](https://arxiv.org/abs/2605.14220).
32. Ziegler, D. M. et al. “Fine-Tuning Language Models from Human Preferences.”
    *arXiv*, 2019. [1909.08593](https://arxiv.org/abs/1909.08593).

</div>
