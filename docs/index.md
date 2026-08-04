---
layout: technical
title: The Policy Triangle
short_title: The Policy Triangle
subtitle: A Unified View of Policy-Mismatch Mitigation in LLM Reinforcement Learning
description: A unified view of methods for mitigating mismatch among behavior, proximal, and current policies in LLM reinforcement learning.
author: Huaiyi Zhao
date: 2026-08-04
---

<figure class="hero-figure">
  <img src="{{ '/assets/policy-mismatch-intro.svg' | relative_url }}" alt="Academic overview of asynchronous LLM reinforcement learning: behavior policy mu generates data, proximal policy q supplies behavior correction and an update anchor, and current policy pi receives the gradient. Methods are located by edge, operator, and geometry.">
</figure>

<div class="abstract" markdown="1">

## Abstract

Large-language-model reinforcement learning increasingly separates rollout
generation from optimization. The policy that generated a response can
therefore differ from both the policy that anchors an update and the policy
currently being trained. Existing mitigation methods use rejection, importance
weighting, clipping, and divergence control, but are often presented as
unrelated algorithms. This article places them in one solution space using the
policy triangle. A method is located by three coordinates: the **edge** on which
mismatch is measured, the **operator** that selects, reweights, or shapes an
update, and the **geometry**—token, sequence, group, or distribution—on which
that operator acts. Mapping existing methods into these coordinates separates
shared design choices from method-specific details and exposes open regions of
the mitigation space. The framework is a unified view, not a new optimizer or
convergence claim.

</div>

## 1. Motivation {#motivation}

Policy-gradient training reuses responses after they have been generated.
Classical methods such as REINFORCE, TRPO, and PPO already distinguish sampling
from updating ([Williams, 1992](https://doi.org/10.1007/BF00992696);
[Schulman et al., 2015](https://arxiv.org/abs/1502.05477);
[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). In LLM systems,
however, the separation is unusually large: rollout engines and training
workers may run different model versions, numerical kernels, precisions, and
routing decisions.

### 1.1 Where mismatch comes from

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

These effects create two conceptually different sources of mismatch. First,
data may already disagree with the learner when it arrives because the rollout
policy is stale or numerically different. Second, the trainable policy may move
further while the same data is reused across optimizer steps. A direct ratio
conflates the two; a factorized view keeps their causes visible.

### 1.2 Why Decoupled PPO matters

Standard PPO normally lets one old policy play two roles: it is treated both as
the policy that generated the data and as the proximal anchor used by clipping.
This coupling is natural when rollouts are fresh and optimization starts
immediately. It becomes ambiguous when a large or asynchronous batch mixes
samples from stale workers, replay, or a numerically different inference
backend.

Let \\(\mu\\) denote the policy that actually generated a token and \\(q\\) the
frozen learner policy at the start of an update. If \\(\mu\\) is stale while the
current policy is still close to \\(q\\), an ordinary PPO ratio
\\(\pi_\theta/\mu\\) can already lie outside its clipping band before the current
optimizer has moved very far. The clip then reacts to two effects at once:
rollout staleness and update drift.

Decoupled PPO separates those roles
([Hilton et al., 2022](https://arxiv.org/abs/2110.00641)). The ratio
\\(q/\mu\\) corrects the behavior-to-learner gap, while
\\(\pi_\theta/q\\) controls the update around a proximal anchor. This was
originally motivated by batch-size invariance, but the same separation is
useful for asynchronous LLM training because it creates an explicit interface
between rollout correction and policy optimization.

The separation is not a free solution to stale data. The behavior-correction
weight can be heavy-tailed, token-level correction does not fully repair prefix
distribution shift, and an old advantage estimate may remain unreliable. The
age of \\(q\\) also creates a speed–stability trade-off: a recent anchor usually
clips less and learns faster, whereas an older anchor can act as a stronger
brake on highly stale data. In synchronous training, \\(\mu=q\\), so coupled and
decoupled PPO coincide.

### 1.3 What existing mitigations do

Despite a rapidly expanding vocabulary, most mitigation mechanisms perform one
of three roles:

1. **Select:** reject a token, response, or group judged too far from a
   reference policy.
2. **Reweight:** change how strongly sampled data contributes, usually with an
   importance ratio or a truncated, tapered, or normalized variant.
3. **Constrain:** shape the trainable update with clipping, a trust region, or a
   divergence-based gate.

The same role can act on a token ratio, a sequence statistic, a prompt group,
or a fuller distributional divergence. Decoupled PPO
([Hilton et al., 2022](https://arxiv.org/abs/2110.00641)), rollout correction
([veRL contributors, 2026](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)),
IcePop and KPop ([Ling Team et al., 2025](https://arxiv.org/abs/2510.18855);
[Guo et al., 2026](https://ringtech.notion.site/kpop)), GSPO
([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)), DPPO
([Qi et al., 2026](https://arxiv.org/abs/2602.04879)), TRM
([Li et al., 2025](https://arxiv.org/abs/2512.23075)), and A-3PO
([Li et al., 2025](https://arxiv.org/abs/2512.06547)) choose different
combinations of these recurring decisions.

Named methods often bundle together the source of mismatch, the intervention
role, and the scale at which it operates. Comparing names alone therefore
obscures which design choice actually differs.

> **Position of this note.** The contribution is a coordinate system for
> viewing the full policy-mismatch mitigation solution space. It maps existing
> methods into shared coordinates and uses the unoccupied regions to organize
> possible designs. It is deliberately narrower than a full theory of
> off-policy RL and does not claim a new optimizer, performance result, or
> convergence theorem.

## 2. The policy triangle as a unified solution space {#framework}

The policy triangle is the framework behind the unified view. It separates
three questions that named algorithms often mix together:

1. **Where is mismatch measured?** Choose an edge.
2. **What does the mitigation do?** Choose an operator.
3. **At what scale does it act?** Choose a geometry.

A mitigation method occupies one or more coordinates in this solution space.
Methods that look different may share coordinates; methods with similar names
may act on different edges or have different gradient semantics.

### 2.1 Three policy anchors and their edges

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
    Figure 1. The mitigation solution space. The triangle identifies where
    mismatch is measured; operators identify what is done; geometry identifies
    the scale of the intervention.
  </figcaption>
</figure>

### 2.2 Direct and factorized topologies

**Bypass.** Set \\(q\equiv\mu\\). Then \\(B=1\\) and \\(U=E\\). Statistical correction
and proximal control are compressed onto one direct edge. This avoids a
proximal forward pass but loses causal attribution.

**Decoupled.** Keep a distinct frozen \\(q\\). The learner can correct rollout
mismatch on \\(B\\) while constraining the current update on \\(U\\), typically as
\\(\mathrm W(B)\mathrm C(U)\\).

For PPO clipping, the two constructions can be written side by side. A coupled
objective attaches the shaper to the direct edge:

<div class="equation">
\[
L_{\mathrm{coupled}}
=
\mathbb E_{\mu}
\left[
\min\!\left(
E_t\hat A_t,\,
\operatorname{clip}(E_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]
</div>

Decoupled PPO attaches a detached behavior weight to \\(B\\) and the
differentiable PPO shaper to \\(U\\):

<div class="equation">
\[
L_{\mathrm{decoupled}}
=
\mathbb E_{\mu}
\left[
\operatorname{sg}[B_t]\,
\min\!\left(
U_t\hat A_t,\,
\operatorname{clip}(U_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]
</div>

On the unclipped branch, the effective score-function coefficient is the same:

<div class="equation">
\[
\operatorname{sg}[B_t]\,U_t\hat A_t
\nabla_\theta\log\pi_\theta
=
E_t\hat A_t\nabla_\theta\log\pi_\theta.
\]
</div>

Decoupling therefore does not introduce a different policy gradient in the
linear limit. It changes where the nonlinear trust-region mechanism is
attached: coupled PPO clips total mismatch \\(E\\), whereas Decoupled PPO
corrects \\(B\\) and clips only update drift \\(U\\). The choice of \\(q\\) is
therefore a speed–stability design variable, not merely an accounting device.

The proximal anchor is itself a design variable. A-3PO, for example,
approximates it by interpolating behavior and current log-probabilities to avoid
an additional model forward pass while retaining the factorization \\(E=BU\\)
([Li et al., 2025](https://arxiv.org/abs/2512.06547)).

### 2.3 Mitigation operators: select, reweight, or constrain

For any edge \\(X\in\{B,U,E\}\\), distinguish operations by their gradient
semantics rather than by their names in a particular implementation.

<div class="table-wrap" markdown="1">

| Operator | Definition | Gradient semantics | Mitigation role |
|---|---|---|---|
| Mask \\(\mathrm M\\) | \\(\mathrm M_{g,s}(X,\hat A)\in\{0,1\}\\) | Detached | **Select:** admit or reject a token, sequence, or group using a ratio band, divergence, age, or metadata. |
| Weight \\(\mathrm W\\) | \\(\mathrm W_{g,s}(X)=\operatorname{sg}[f_{g,s}(X)]\ge0\\) | Detached | **Reweight:** change measure or influence using raw, truncated, tapered, or self-normalized importance sampling. |
| Shaper \\(\mathrm C\\) | Enters the loss through \\(\pi_\theta\\) | Differentiable | **Constrain:** reshape the current update using clipping, smooth gates, or divergence geometry. |

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

### 2.4 Correction geometry and tail operators

Token/sequence and IS/TIS/MIS/RS answer different questions. The first pair
chooses **which probability object is measured**; the second chooses **what is
done with its tail**. Treating these as two independent axes makes the
terminology in the analyses of
[Li and Liu, 2025a](https://richardli.xyz/post/rl-collapse-part2/) and
[2025b](https://richardli.xyz/post/rl-collapse-part3/) much easier to compare.

Let the behavior-edge token ratio be
\\(r_t\equiv B_t=q(y_t\mid h_t)/\mu(y_t\mid h_t)\\). In a bypass topology, the
same discussion applies with \\(r_t=E_t\\). Let
\\(\phi_t=\hat A_t\nabla_\theta\log\pi_\theta(y_t\mid h_t)\\) denote a local
gradient contribution and \\(F(y)\\) a response-level contribution.

#### First choice: token, prefix, sequence, or geometric statistic

The four commonly used statistics are

<div class="equation">
\[
r_t=B_t,\qquad
R_{1:t}=\prod_{j=1}^{t}r_j,\qquad
R=R_{1:T}=\prod_{t=1}^{T}r_t,\qquad
G=R^{1/T}=\exp\!\left(\frac1T\sum_t\log r_t\right).
\]
</div>

They do not have interchangeable semantics:

<div class="table-wrap" markdown="1">

| Geometry | What it changes or measures | Bias–variance consequence |
|---|---|---|
| Token \\(r_t\\) | Corrects the sampled action conditional on the observed prefix \\(h_t\\). | Local and relatively low variance, but leaves the prefix or state-occupancy distribution under \\(\mu\\); biased relative to a full \\(q\\)-trajectory objective. |
| Prefix \\(R_{1:t}\\) | Changes measure for a causal contribution at position \\(t\\), without ratios from future tokens. | Can be exact for the corresponding per-decision objective, but variance grows along the prefix and depends on the return/advantage construction. |
| Sequence \\(R\\) | Changes the complete response measure: \\(R=q(y\mid x)/\mu(y\mid x)\\). | Untruncated sequence IS is exact under support and common-dynamics assumptions, but its second moment and weight concentration can grow rapidly with length. |
| Geometric \\(G\\) | Measures average sampled log-ratio per token. | Length normalized and useful for gating or diagnostics, but **not** a density ratio and therefore not an IS weight. Opposite-signed token log-ratios can also cancel. |

</div>

For sequence IS, the change-of-measure identity is

<div class="equation">
\[
\mathbb E_{y\sim\mu}[R(y)F(y)]
=
\mathbb E_{y\sim q}[F(y)].
\]
</div>

Its statistical problem is not merely that \\(R\\) can be numerically large.
Because \\(\log R=\sum_t\log r_t\\), dispersion accumulates across the
response; a few trajectories can dominate the batch and drive effective sample
size toward one. Fixed sequence thresholds are also length sensitive because
the distribution of a sum of log-ratios changes with \\(T\\). Token IS avoids
the product, but it pays for that stability by not correcting how the prefix
itself was generated.

<figure class="triangle-figure">
  <img src="{{ '/assets/is-tail-operators.svg' | relative_url }}" alt="Academic diagram separating importance-sampling geometry from tail operators. Token, prefix, sequence, and geometric statistics feed into IS, TIS, MIS, and rejection sampling. IS keeps the raw ratio, TIS caps it, MIS masks the tail while retaining the IS weight, and rejection sampling applies a pure selection mask.">
  <figcaption>
    Figure 2. Rollout correction is the composition of a ratio statistic and a
    tail operator. The same operator can act at token or sequence scale; the
    geometric mean is a length-normalized gating statistic, not an IS weight.
  </figcaption>
</figure>

#### Second choice: IS, TIS, MIS, or RS

For a valid density ratio \\(R\\), a cap \\(C\\), and an acceptance statistic
\\(S\\), the four operators can be separated algebraically:

<div class="equation">
\[
\begin{aligned}
\text{IS:}\;& RF,\\
\text{TIS:}\;& \min(R,C)F,\\
\text{MIS:}\;& \mathbf 1\{R\le C\}\,RF,\\
\text{RS:}\;& \mathbf 1\{S\in\mathcal A\}\,F.
\end{aligned}
\]
</div>

**IS** keeps the raw change-of-measure weight. **Truncated IS (TIS)** caps a
large weight but keeps the sample, treating the tail as informative but too
noisy. **Masked IS (MIS)** drops the tail and retains the original IS weight
inside the accepted region: a useful mnemonic is **MIS = mask × IS**.
In the rollout-correction terminology used here, **rejection sampling (RS)** is
pure selection: accepted data receives whatever base loss or separately
configured weight follows the mask. This is narrower than the general
statistical use of “rejection sampling.” The mask may use a one-sided ratio
threshold, a two-sided band, a geometric statistic, or a divergence.

This distinction resolves an implementation-level naming ambiguity. In the
[veRL rollout-correction formulation](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html),
RS modifies the response mask and can be composed with a separate IS choice.
Thus Seq-MIS, \\(\mathbf 1\{R\le C\}RF\\), is one particular composition; it is
not a synonym for every sequence-level rejection rule. The pure geometric rule
called Geo-Mask in Li and Liu's Part 3, and Geo-RS in veRL, instead has the form
\\(\mathbf 1\{C_{\rm low}\le G\le C_{\rm high}\}F\\). It performs selection but
no change of measure.

At token scale, replace \\(R,F\\) by \\(r_t,\phi_t\\). The resulting matrix makes
the combinations explicit:

<div class="table-wrap" markdown="1">

| Scale | IS | TIS | MIS | Pure RS |
|---|---|---|---|---|
| Token | \\(r_t\phi_t\\) | \\(\min(r_t,C)\phi_t\\) | \\(\mathbf1\{r_t\le C\}r_t\phi_t\\) | \\(\mathbf1\{r_t\in\mathcal A_t\}\phi_t\\) |
| Sequence | \\(RF\\) | \\(\min(R,C)F\\) | \\(\mathbf1\{R\le C\}RF\\) | \\(\mathbf1\{R\in\mathcal A\}F\\) |
| Geometric gate | Not valid: \\(G\\) is not a density ratio | A heuristic shaper, not TIS in the change-of-measure sense | \\(\mathbf1\{G\in\mathcal A\}RF\\): geometric mask plus sequence IS | \\(\mathbf1\{G\in\mathcal A\}F\\): Geo-RS |

</div>

The matrix gives algebraic descriptions, not universally standardized method
names. In particular, “Token-MIS” and “Seq-RS” may be labeled differently by
different codebases; the formula should take precedence over the acronym.

#### Reading the bias–variance trade-off

The operator determines how the tail is treated; the geometry determines what
kind of mismatch remains:

1. **Seq-IS:** no truncation or selection bias in the trajectory
   change-of-measure identity, but potentially catastrophic weight variance and
   poor effective sample size for long responses.
2. **Token-IS:** much lower variance under bounded per-token terms, but prefix
   or occupancy bias remains even before any truncation is applied.
3. **TIS:** caps tail influence and usually lowers variance, while adding
   \\(\mathbb E_\mu[(\min(R,C)-R)F]\\) truncation bias. Seq-TIS broadcasts one
   capped response weight to every token; Token-TIS adds truncation bias on top
   of the token approximation.
4. **MIS:** bounds the accepted IS weight and removes suspected bad-tail data,
   but sacrifices sample efficiency. For Seq-MIS its bias is exactly the omitted
   target-tail contribution, \\(-\mathbb E_q[F\mathbf1\{R>C\}]\\).
5. **RS:** can reject on a more robust or length-normalized statistic, but is a
   selection mechanism rather than off-policy correction. Geo-RS avoids a raw
   product threshold's length scale, yet can miss localized token outliers or
   cancellation; it is often paired with Token-TIS to obtain a sequence gate
   plus local weights.

These rankings require qualifications. Bounds such as polynomial token-level
variance or capped sequence-level variance assume bounded score and
return/advantage terms; stale or incorrect advantages can still dominate. A
fixed threshold can also create length-conditioned truncation or rejection, so
ESS and acceptance rates should always be reported by response length.
Finally, detached TIS is not PPO clipping: TIS caps a statistical weight,
whereas PPO uses an advantage-dependent differentiable shaper as described in
Section 2.3.

A broad class of critic-free policy-loss gradients can now be read as an
explicit composition:

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

### 2.5 Combining mitigation mechanisms

The factorization \\(E=BU\\) does not make interventions interchangeable. Where
an operator is attached determines what mismatch it sees, how often it must be
recomputed, and which bias–variance trade-off it introduces.

#### Nonlinear placement matters

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

#### Normalization after truncation

After an IS geometry and truncation rule have been chosen, self-normalization
is an orthogonal variance-control decision:

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

#### A bookkeeping sanity check

When an estimator is specifically intended to reproduce the direct
behavior-to-current change-of-measure kernel, its unshaped limit should reduce
to \\(E=BU\\). This conditional check can catch accidental duplication: for
example, \\(\mathrm W(E)\mathrm C(E)\\) reduces to \\(E^2\\), whereas the factorized
construction \\(\mathrm W(B)\mathrm C(U)\\) reduces to \\(BU\\).

This is an implementation diagnostic, not a correctness or unbiasedness
criterion. Clipping, masking, geometric aggregation, and self-normalization may
deliberately move an estimator away from that limit.

### 2.6 Mapping existing methods into the solution space {#taxonomy}

The table maps representative methods by topology and by their dominant
solution-space coordinates. It isolates policy-mismatch mitigation and does not
enumerate reward shaping, advantage construction, entropy bonuses, dynamic
sampling, or systems optimizations.

<div class="table-wrap" markdown="1">

| Method family | Topology | Solution-space coordinate | Mitigation role |
|---|---|---|---|
| [PPO](https://arxiv.org/abs/1707.06347) / [GRPO](https://arxiv.org/abs/2402.03300) / [DAPO](https://arxiv.org/abs/2503.14476) | Coupled | \\(\mathrm C_{\mathrm{tok}}(E)\\) | One direct ratio carries correction and proximal-control roles; DAPO uses asymmetric bounds. |
| [CISPO](https://arxiv.org/abs/2506.13585) / [TOPR](https://arxiv.org/abs/2503.14286) | Direct PG | \\(\mathrm W_{\mathrm{tok}}(E)\\) | Detached clipped or tapered weight; gradients need not vanish outside a PPO band. |
| [GSPO](https://arxiv.org/abs/2507.18071) | Coupled | \\(\mathrm C_{\mathrm{geo}}(E)\\) | Sequence-geometric shaping; the original objective does not add a separate pre-filter. |
| [Decoupled PPO](https://arxiv.org/abs/2110.00641) / [AReaL](https://arxiv.org/abs/2505.24298) / [A-3PO](https://arxiv.org/abs/2512.06547) | Factorized | \\(\mathrm W(B)\mathrm C(U)\\) | Behavior correction is detached from the trainable proximal constraint. |
| [TIS](https://doi.org/10.1198/106186008X320456) / rollout IS | Either | \\(\mathrm W_{\mathrm{tok/seq}}(B\text{ or }E)\\) | Raw or truncated detached weights trade change-of-measure fidelity for variance control. |
| MIS | Either | \\(\mathrm M_{\mathrm{tok/seq}}(B\text{ or }E)\mathrm W_{\mathrm{raw}}(B\text{ or }E)\\), optional \\(\mathrm C\\) | A hard mask removes the tail; accepted samples retain their raw IS weight. |
| RS / Geo-RS | Either | \\(\mathrm M_{\mathrm{tok/seq/geo}}(B\text{ or }E)\\), optional separate \\(\mathrm W\\) and \\(\mathrm C\\) | Pure selection is orthogonal to weighting; a geometric gate is length normalized but is not an IS ratio. |
| [IcePop](https://arxiv.org/abs/2510.18855) / [KPop](https://ringtech.notion.site/kpop) | Usually direct | \\(\mathrm M_{\mathrm{tok}}(E)\\) plus a base loss | IcePop uses a ratio region; KPop uses bidirectional binary-KL geometry sensitive to absolute probability. |
| [DPPO](https://arxiv.org/abs/2602.04879) / [TRM](https://arxiv.org/abs/2512.23075) | Direct | Divergence \\(\mathrm M(E)\\) or \\(\mathrm C(E)\\) | Distributional geometry replaces sampled-ratio magnitude; TRM rejects a sequence on worst-token divergence. |

</div>

#### What the mapping clarifies

1. **Bypass is not “IS disabled.”** It means \\(B=1\\); the direct loss may still
   contain \\(E=U\\).
2. **PPO and GSPO clipping are not pre-filters.** Their branches change gradient
   flow in an advantage-dependent direction; a detached mask rejects
   independently of that direction.
3. **A decoupled filter may read any raw edge.** \\(\mathrm M(B)\\) is static when
   \\(q\\) is frozen, whereas \\(\mathrm M(U)\\) and \\(\mathrm M(E)\\) generally
   change as \\(\pi\\) is updated.

#### Relation to nearby views

[Decoupled PPO](https://arxiv.org/abs/2110.00641) is the closest precursor
because it explicitly inserts a proximal policy between behavior and current
policies. [AReaL](https://arxiv.org/abs/2505.24298) and
[A-3PO](https://arxiv.org/abs/2512.06547) operationalize or approximate that
factorization for asynchronous LLM training. The policy triangle broadens that
factorization into an edge–operator–geometry solution space without prescribing
a particular proximal anchor.

The [veRL rollout-correction formulation](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
and [FP8-RL](https://arxiv.org/abs/2601.18150) compare practical TIS, MIS,
rejection, bypass, and decoupled configurations. The triangle abstracts that
implementation pipeline and additionally places direct PPO, GSPO, and
divergence-based shapers in the same coordinates.

[RPG](https://arxiv.org/abs/2505.17508) and the
[off-policy interpretation of group-relative REINFORCE](https://arxiv.org/abs/2509.24203)
give deeper analyses of KL regularization, baselines, and data shaping. Those
axes are intentionally outside this critic-free mitigation view.
[Jackpot](https://arxiv.org/abs/2602.06107) also lies partly outside the view
because it changes the behavior distribution through rejection and joint
rollout-model updates, rather than only operating on stored trajectories
downstream.

## 3. Implications and open regions in the solution space {#implications}

Once mitigation methods are separated into edge, operator, and geometry, their
design trade-offs become easier to state. The framework does not prescribe one
best method; it shows which problem a choice is trying to solve and which
alternatives remain available.

### 3.1 Reading mitigation choices from their coordinates

| Observed issue | Relevant coordinate | Design implication |
|---|---|---|
| Stale rollouts or train–inference disagreement | Behavior edge \\(B\\) | Use a detached mask or weight to select or correct data that has already been generated. |
| Drift during repeated optimizer updates | Update edge \\(U\\) | Use a differentiable shaper or dynamic trust-region test tied to the current policy. |
| Both sources are small or cannot be separated operationally | Direct edge \\(E\\) | A coupled method is simpler, but gives up causal attribution between admission mismatch and update drift. |
| A few tokens dominate the discrepancy | Token or distributional geometry | Use local ratio or divergence statistics rather than rejecting an entire response. |
| Coherence of the whole response matters | Sequence geometry | Use sequence-product, geometric, worst-token, or sequence-divergence statistics according to the intended semantics. |

A useful default is therefore to place detached statistical correction near the
behavior anchor and differentiable control near the current anchor. This is a
design heuristic, not a theorem. A direct method may be preferable when
staleness is low, an extra proximal forward pass is expensive, or the system
does not retain a distinct \\(q\\).

### 3.2 Open regions suggested by the framework

The solution space is combinatorial: an edge choice does not determine an
operator or geometry. Existing methods occupy only some combinations. The
following examples illustrate open regions rather than claim new named
algorithms or expected improvements:

1. **Decoupled sequence shaping:**
   \\(\mathrm W_{\mathrm{tok}}(B)\mathrm C_{\mathrm{geo}}(U)\\). Correct rollout
   mismatch token-wise, then constrain the update with a sequence-geometric
   proximal ratio.
2. **Heterogeneous two-edge mitigation:** apply entropy-adaptive admission and
   \\(\mathrm W(B)\\) on the behavior edge, followed by a divergence-based mask
   or shaper on \\(U\\). Each edge is treated with geometry appropriate to its
   source.
3. **Segment-aware asynchronous mitigation:** normalize or truncate
   \\(\mathrm W(B_t)\\) within policy-version segments, optionally read total
   mismatch \\(E\\) with a sequence gate, and retain \\(\mathrm C(U)\\) for the
   current update.

These coordinates are hypotheses to test. Their value here is to make the
unexplored design choices explicit, not to imply that more elaborate
compositions will outperform simpler ones.

### 3.3 Implications for asynchronous and piecewise rollouts

A resumed or interrupted trajectory may be generated by several policy
versions. Let \\(v(t)\\) denote the version used at token \\(t\\):

<div class="equation">
\[
\mu(y\mid x)=\prod_t\mu_{v(t)}(y_t\mid h_t),\qquad
B_t=\frac{q_t}{\mu_{v(t),t}}.
\tag{7}
\]
</div>

The framework treats policy version as metadata attached to the behavior edge,
not as a new topology. Stored per-token sampling log-probabilities support the
observed conditional ratio, but do not correct unlogged prompt selection,
replay selection, or resume-selection bias. Segment-aware normalization is
therefore a geometry and normalization choice within the same solution space.

### 3.4 Using the framework to analyze a method

1. **Identify the actual behavior distribution.** Include sampling transforms,
   backend precision, and piecewise versions.
2. **Identify the proximal anchor.** State whether it equals behavior, is a
   frozen snapshot, or is approximated.
3. **Locate every intervention.** Record its edge, operator, statistic,
   geometry, sign rule, and normalization domain.
4. **Inspect compositions.** Distinguish deliberate bias–variance choices from
   accidental missing or duplicated ratio factors.
5. **Measure the coordinates separately.** Report token \\(\log B_t\\),
   cumulative prefix and sequence log-ratios, \\(U\\), and \\(E\\). Track
   effective sample size before and after truncation, truncation or rejection
   rates by response length, gradient variance, policy age, reward, and
   throughput.

## 4. Limitations and conclusion {#scope}

The policy triangle organizes the mitigation solution space; it does not rank
its coordinates. It cannot choose thresholds, guarantee that masking improves
return, repair support mismatch, or replace empirical validation. It omits
value/advantage off-policy correction, reward-model drift, environment
nonstationarity, and optimizer-state staleness. Sequence-product importance
sampling can be formally exact and statistically unusable, while geometric
statistics can be stable without being exact change-of-measure factors.

Within this scope, the behavior, proximal, and current policies provide a
compact set of anchors. Masks select regions, detached weights change measure,
and differentiable shapers control the update; each chooses its own geometry.
Bypass collapses the triangle, while decoupling separates exogenous rollout
mismatch from endogenous update drift.

The main value is a clearer view of the whole solution space. The
edge–operator–geometry representation separates **where mismatch is measured**,
**what a mitigation mechanism does**, and **at what scale it acts**. Existing
methods become recognizable combinations of shared choices rather than a list
of unrelated losses. The same coordinates also expose open regions that can be
tested without presenting them as established improvements.

### Suggested citation {#citation}

    @misc{zhao2026policytriangle,
      title  = {The Policy Triangle: A Unified View of Policy-Mismatch
                Mitigation in LLM Reinforcement Learning},
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
33. Li, Y., and Liu, J. “Part 2: Applying the SGA Framework—Token vs.
    Sequence-Level Correction.” Blog post, 2025.
    [Article](https://richardli.xyz/post/rl-collapse-part2/).
34. Li, Y., and Liu, J. “Part 3: Trust Region Optimization via Sequence
    Masking.” Blog post, 2025.
    [Article](https://richardli.xyz/post/rl-collapse-part3/).

</div>
