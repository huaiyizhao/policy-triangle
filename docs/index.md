---
layout: technical
title: The Policy Triangle
short_title: The Policy Triangle
subtitle: A Unified View of Policy-Mismatch Mitigation in LLM Reinforcement Learning
description: A unified view of methods for mitigating mismatch among behavior, proximal, and current policies in LLM reinforcement learning.
author: Huaiyi Zhao
date: 2026-08-18
---

<figure class="hero-figure">
  <img src="{{ '/assets/policy-mismatch-intro.svg' | relative_url }}" alt="Academic graphical abstract showing the progression from the exact off-policy policy gradient to the local q-surrogate and its decoupled approximation surfaces: behavior correction B from mu to q, credit realignment toward A q, and proximal update U from q to pi. Existing LLM methods concentrate on B and U, while explicit credit realignment remains comparatively underexplored.">
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
that operator acts. A fourth label, the **credit anchor**, records whether the
advantage is generated under the behavior policy, aligned to the proximal
policy, or simply assumed to be close. Mapping existing methods into these
coordinates separates shared design choices from method-specific details and
exposes open regions of the mitigation space. The framework is a unified view
of approximations to a proximal-policy surrogate, not a new optimizer,
unbiasedness result, or convergence claim. The main text emphasizes
interpretive conclusions; a technical appendix records the derivations and
exactness conditions.

</div>

## 1. Motivation {#motivation}

Policy-gradient training reuses responses after they have been generated.
Classical methods already separate sampling from updating, but the successive
steps from exact policy gradient to TRPO, PPO, and Decoupled PPO also change
what objective is being estimated. Making that evolution explicit gives the
policy triangle a precise target: most methods studied below are estimators or
robust approximations of a proximal-policy surrogate, not unbiased estimators
of the current-policy gradient at an arbitrary \\(\pi\\).

In plain language, one model may write the response, a frozen snapshot may
define what counts as a safe update, and a newer model may receive that update.
A method called “rollout correction” can address one of these gaps without
addressing the others. The purpose of this article is to make those roles
visible before comparing algorithm names.

### 1.1 Exact off-policy policy gradient and its two obstacles

Let \\(\mu\\) be the policy that generated a response, \\(\pi=\pi_\theta\\) the
current policy, and \\(h_t=(x,y_{\lt t})\\) the token context. Write
\\(E_i=\pi(y_i\mid h_i)/\mu(y_i\mid h_i)\\) and
\\(E_{1:t}=\prod_{i=1}^tE_i\\).

Under the usual support and common-dynamics assumptions, the exact
current-policy gradient has the change-of-measure form

<div class="equation">
\[
\nabla J(\pi)
=
\sum_{t=1}^{T}
\mathbb E_{y\sim\mu}
\left[
E_{1:t}A_t^\pi(h_t,y_t)
\nabla_\theta\log\pi_\theta(y_t\mid h_t)
\right].
\]
</div>

This identity exposes two different statistical obstacles. The prefix ratio
\\(E_{1:t}\\) corrects how the context and sampled action were reached, but its
product can have prohibitive variance. The credit term must meanwhile be
\\(A_t^\pi\\): the continuation after \\(y_t\\) must be evaluated under \\(\pi\\),
not under the policy that happened to generate the recorded suffix. Full
sequence IS performs both corrections at once, but is often statistically
unusable for long responses. The exact expression is therefore a benchmark
for what is being approximated, not a default recipe for LLM training
([Williams, 1992](https://doi.org/10.1007/BF00992696)).

> **In plain language.** Exact correction has two jobs: correct how the model
> reached the current token, and evaluate what happens after that token under
> the policy we actually care about. Long products make the first job noisy;
> unavailable current-policy continuations make the second job difficult. The
> terminal-return identity and change-of-measure details are collected in
> [Appendix A.1](#appendix-a1).

The gap is unusually visible in LLM systems because rollout and training are
usually separate ([Sheng et al., 2024](https://arxiv.org/abs/2409.19256);
[Fu et al., 2025](https://arxiv.org/abs/2505.24298)). Stale weights, different
attention kernels or precisions, and mixture-of-experts routing can make the
actual rollout distribution disagree with the learner even when both
nominally use the same checkpoint
([Qiu et al., 2026](https://arxiv.org/abs/2601.18150);
[Zhong et al., 2026](https://arxiv.org/abs/2605.14220)). Reusing a response
across optimizer steps adds a second source of drift
([Zhang et al., 2026](https://arxiv.org/abs/2602.01826)).

### 1.2 TRPO replaces the exact gradient with a local improvement model

Let \\(q\\) be a frozen reference policy. Instead of correcting all the way to
an arbitrary current policy, TRPO asks a more practical question: *what update
looks beneficial if \\(\pi\\) stays close to \\(q\\)?* Its conclusion is the
local surrogate

<div class="equation">
\[
L_q^{\mathrm{TRPO}}(\pi)
=
J(q)+
\sum_{t=1}^{T}
\mathbb E_{h_t\sim d_t^q,\,y_t\sim q}
\left[
U_tA_t^q(h_t,y_t)
\right],
\qquad
U_t=\frac{\pi(y_t\mid h_t)}{q(y_t\mid h_t)}.
\]
</div>

The reference advantage \\(A^q\\) is not an ad hoc substitution: it belongs to
the exact performance-difference identity. The approximation is to reuse the
context distribution of \\(q\\). The surrogate and true objective agree to
first order at \\(q\\), but their gradients generally differ away from it. TRPO
controls that local-model error with an explicit trust-region constraint rather
than trying
to estimate the high-variance exact gradient everywhere
([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)). The exact identity
and first-order matching statement appear in [Appendix A.2](#appendix-a2).

### 1.3 PPO keeps the local surrogate and replaces the trust-region solver

PPO retains the same \\(q\\)-anchored local model but replaces TRPO's constrained
optimization with an advantage-dependent clipped objective:

<div class="equation">
\[
L_q^{\mathrm{PPO}}(\pi)
=
\mathbb E_q
\left[
\sum_{t=1}^{T}
\min\!\left(
U_t\hat A_t^q,
\operatorname{clip}(U_t,1-\epsilon,1+\epsilon)\hat A_t^q
\right)
\right].
\]
</div>

At \\(\pi=q\\), clipping is inactive and the gradient matches the on-policy
gradient at \\(q\\). Away from that point, PPO is neither the exact
current-policy gradient nor the unclipped TRPO surrogate. The clip is better
understood as a heuristic differentiable shaper that discourages selected
directions of departure from \\(q\\); it does not reproduce TRPO's strict
constraint or its monotonic-improvement argument
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). Both methods
therefore rely on keeping \\(\pi\\) sufficiently close to the reference at which
their local model is accurate.

> **In plain language.** TRPO says “optimize only inside a measured
> neighborhood.” PPO keeps the same local picture but uses clipping as a
> cheaper guardrail. The guardrail is useful, but it is not the original TRPO
> guarantee.

### 1.4 Decoupled PPO exposes three approximation surfaces

Standard PPO normally lets one distribution play two roles: it is treated as
both the behavior policy and the proximal anchor. In an asynchronous pipeline,
let \\(\mu\\) denote the distribution that actually generated the token, \\(q\\)
the frozen proximal policy, and \\(\pi\\) the current trainable policy. Define

<div class="equation">
\[
B_t=\frac{\color{#c66d0a}{q_t}}{\color{#2878ad}{\mu_t}},
\qquad
U_t=\frac{\color{#19845d}{\pi_{\theta,t}}}{\color{#c66d0a}{q_t}},
\qquad
E_t=\frac{\color{#19845d}{\pi_{\theta,t}}}{\color{#2878ad}{\mu_t}}=B_tU_t.
\]
</div>

Using \\(\mu\\)-data, the exact gradient of the *unclipped* \\(q\\)-surrogate is

<div class="equation key-equation">

<div class="equation-label">Decoupled PPO reference gradient</div>

\[
\begin{aligned}
g_{\color{#c66d0a}{q}}(\color{#19845d}{\pi_\theta})
={}&
\sum_{t=1}^{T}
\mathbb E_{y\sim\color{#2878ad}{\mu}}
\Bigg[
\underbrace{
\prod_{i=1}^{t}
\frac{\color{#c66d0a}{q}(y_i\mid h_i)}
     {\color{#2878ad}{\mu}(y_i\mid h_i)}
}_{B_{1:t}\;\text{— past/data}}
\\[-2pt]
&\qquad\times
\underbrace{
\frac{\color{#19845d}{\pi_\theta}(y_t\mid h_t)}
     {\color{#c66d0a}{q}(y_t\mid h_t)}
}_{U_t\;\text{— current update}}
\underbrace{A_t^{\color{#c66d0a}{q}}}_{\text{future/credit}}
\nabla_\theta\log
\color{#19845d}{\pi_\theta}(y_t\mid h_t)
\Bigg].
\end{aligned}
\]

<div class="policy-legend" aria-label="Policy color legend">
  <span class="policy-mu"><span class="policy-symbol">μ</span> behavior / rollout</span>
  <span class="policy-q"><span class="policy-symbol">q</span> frozen proximal anchor</span>
  <span class="policy-pi"><span class="policy-symbol">π<sub>θ</sub></span> current trainable policy</span>
</div>

</div>

Read the highlighted expression from left to right: \\(B_{1:t}\\) repairs the
history inherited from rollout, \\(A_t^q\\) says how the sampled action should
be credited under the proximal policy, and \\(U_t\\) controls the trainable
step from \\(q\\) to \\(\pi\\). If only a behavior-generated terminal return is
available, future ratios can in principle reconstruct the missing
\\(q\\)-credit; the corresponding identity is in
[Appendix A.3](#appendix-a3).

This yields three conceptually separate design problems:

1. **Past or data correction:** approximate \\(B_{1:t}\\) without letting a
   prefix product dominate the batch.
2. **Future or credit realignment:** obtain \\(A^q\\) from behavior-generated
   returns, a critic, corrected suffixes, or another estimator.
3. **Proximal optimization:** shape \\(U\\) while \\(\pi\\) remains near \\(q\\).

The research coverage of these three surfaces is uneven. Among the LLM
policy-loss methods mapped in this article, explicit machinery is concentrated
on \\(B\\) and \\(U\\). Comparatively few methods construct a dedicated
\\(A^{q\leftarrow\mu}\\) estimator; most reuse behavior-generated rollout
credit and assume \\(\hat A^{\mathrm{roll}}\approx A^q\\). Critics, suffix
correction, trace estimators, and re-rolled continuations are possible
credit-realignment tools, but they are not yet a comparably mature design axis
in this scoped literature. This is a statement about the methods surveyed here,
not a claim that off-policy advantage estimation is absent from reinforcement
learning.

Decoupled PPO makes the first and third problems explicit
([Hilton et al., 2022](https://arxiv.org/abs/2110.00641)). Most LLM variants
then simplify \\(B_{1:t}\\) to a token or sequence statistic, clip or mask its
tail, retain PPO-like shaping on \\(U\\), and reuse a behavior-generated credit
signal as if it were \\(A^q\\). The factorization is exact algebra; the resulting
loss need not be an exact estimator of either the \\(q\\)-surrogate or
\\(\nabla J(\pi)\\).

The remainder therefore stays focused on how practical methods approximate or
robustify this Decoupled-PPO \\(q\\)-surrogate; correcting a rollout ratio does
not by itself remove the surrogate's local-model error or realign its credit.

### 1.5 The solution space studied here

The rest of this article uses the Decoupled-PPO \\(q\\)-surrogate as its default
reference estimand and treats existing mitigations as alternative estimators or
robust approximations around it. A direct full-sequence estimator that instead
targets the exact current-policy gradient is marked as such rather than being
silently identified with the surrogate. The methods repeatedly make four
choices:

1. **Edge:** act on behavior mismatch \\(B\\), update drift \\(U\\), or their
   collapsed product \\(E=BU\\).
2. **Operator:** select with a mask, reweight with a detached coefficient, or
   shape a differentiable update.
3. **Geometry:** use a token, prefix, sequence, group, sampled ratio, or fuller
   distributional statistic.
4. **Credit anchor:** use rollout-generated credit, explicitly estimate
   \\(A^q\\), or assume the two are close.

Rollout correction, TIS, MIS, and rejection sampling explore different
approximations on the behavior or direct edge
([veRL contributors, 2026](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)).
IcePop and KPop alter admission and weighting on \\(B\\)
([Ling Team et al., 2025](https://arxiv.org/abs/2510.18855);
[Guo et al., 2026](https://ringtech.notion.site/kpop)); GSPO changes the update
geometry ([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)); DPPO and TRM
use distributional mismatch ([Qi et al., 2026](https://arxiv.org/abs/2602.04879);
[Li et al., 2025](https://arxiv.org/abs/2512.23075)); and A-3PO approximates the
proximal anchor ([Li et al., 2025](https://arxiv.org/abs/2512.06547)). They are
not unrelated losses: they occupy different approximation coordinates around
the same local-surrogate structure.

> **Position of this note.** The policy triangle is a coordinate system for
> this surrogate-estimation solution space. It distinguishes exact raw IS,
> biased variance-control operators, and differentiable optimizer shaping; it
> does not claim that every mapped method is an unbiased estimator of the
> current-policy gradient, nor does it introduce a new optimizer or convergence
> theorem.

## 2. The policy triangle as a unified solution space {#framework}

The policy triangle is the framework behind the unified view. Unless stated
otherwise, the reference estimand is the Decoupled-PPO \\(q\\)-surrogate
highlighted in Section 1.4. A useful mental model is a three-stage pipeline:
**admit or correct old data**, **assign compatible credit**, then **shape the
new update**.

A mapped method may estimate this expression exactly, approximate one of its
factors, or deliberately replace it with a more robust masked or shaped
objective. The framework separates four questions that named algorithms often
mix together:

1. **Where is mismatch measured?** Choose an edge.
2. **What does the mitigation do?** Choose an operator.
3. **At what scale does it act?** Choose a geometry.
4. **Which policy supplies credit?** State the advantage anchor.

A mitigation method occupies one or more coordinates plus a credit label.
Methods that look different may share coordinates; methods with similar names
may act on different edges, assume different advantages, or have different
gradient semantics.

### 2.1 Policy anchors, edges, and operators

For prompt \\(x\\), response \\(y=(y_1,\ldots,y_T)\\), and token context
\\(h_t=(x,y_{\lt t})\\), write the sampled-token probabilities as
\\(\mu_t\\), \\(q_t\\), and \\(\pi_t\\) under the behavior, proximal, and current
policies respectively.

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
| Behavior edge | \\(B=q/\mu\\) | Past/data mismatch already present at learner admission; raw prefix products correct occupancy under \\(\mu\\) to occupancy under \\(q\\). |
| Update edge | \\(U=\pi/q\\) | Drift created by the current optimization step relative to its proximal anchor. |
| Direct edge | \\(E=\pi/\mu=BU\\) | Total behavior-to-current mismatch after the two roles have been collapsed. |

The ideal surrogate also requires a compatible credit signal. The edge
factorization alone does not turn a return generated by a \\(\mu\\)-suffix into
\\(A_t^q\\):

<div class="equation">
\[
\underbrace{B_{1:t}}_{\text{past correction}}
\quad
\underbrace{A_t^{q\leftarrow\mu}}_{\text{future credit realignment}}
\quad
\underbrace{U_t}_{\text{proximal update}}.
\]
</div>

Here \\(A_t^{q\leftarrow\mu}\\) denotes an estimator intended to target
\\(A_t^q\\) using behavior-generated data. It may use suffix IS
\\(B_{t+1:T}\\), a \\(q\\)-critic, a trace estimator, re-rolled continuations, or
no explicit correction at all. In the last case the method uses a rollout
credit signal \\(\hat A^{\mathrm{roll}}\\) and implicitly assumes
\\(\hat A^{\mathrm{roll}}\approx A^q\\). For GRPO-style group normalization,
\\(\hat A^{\mathrm{roll}}\\) is safer terminology than literally calling the
response-level signal a token-level \\(A^\mu\\).

The three operator classes differ by gradient semantics rather than by the
names used in a particular implementation.

<div class="table-wrap" markdown="1">

| Operator | Definition | Gradient semantics | Mitigation role |
|---|---|---|---|
| Mask \\(\mathrm M\\) | \\(\mathrm M_{g,s}(X,\hat A)\in\{0,1\}\\) | Detached | **Select:** admit or reject a token, sequence, or group using a ratio band, divergence, age, or metadata. |
| Weight \\(\mathrm W\\) | \\(\mathrm W_{g,s}(X)=\operatorname{sg}[f_{g,s}(X)]\ge0\\) | Detached | **Reweight:** change measure or influence using raw, truncated, tapered, or self-normalized importance sampling. |
| Shaper \\(\mathrm C\\) | Enters the loss through \\(\pi_\theta\\) | Differentiable | **Constrain:** reshape the current update using clipping, smooth saturation, a divergence penalty, or a trust-region geometry. |

</div>

Parameter dependence gives the edges a natural, but not exclusive, division of
labor:

<div class="table-wrap" markdown="1">

| Edge | Dependence during an update | Canonical attachments |
|---|---|---|
| \\(B=q/\mu\\) | Frozen when \\(q\\) and \\(\mu\\) are fixed | \\(\mathrm M(B),\mathrm W(B)\\): admission and behavior correction. A standalone \\(\mathrm C(B)\\) has zero gradient; used as a coefficient, it is a detached weight. |
| \\(U=\pi/q\\) | Changes with \\(\theta\\) | \\(\mathrm C(U)\\): proximal update shaping. Detached \\(\mathrm M(U)\\) or \\(\mathrm W(U)\\) are also possible, but become dynamic update-drift rules. |
| \\(E=\pi/\mu\\) | Changes with \\(\theta\\) and conflates both gaps | \\(\mathrm M(E),\mathrm W(E),\mathrm C(E)\\) are all available, at the cost of losing attribution between rollout mismatch and update drift. |

</div>

The canonical factorized composition is therefore

<div class="equation">
\[
\boxed{
\mathrm M(B)\,\mathrm W(B)\,
\mathrm C\!\left(U;\hat A^{q\leftarrow\mu}\right)
}
\qquad
\text{correct past data on }B,\text{ supply }q\text{-credit, shape }U.
\]
</div>

Here either detached operator may be the identity when no filtering or
reweighting is required. The credit label is not a fourth multiplicative ratio:
it records which continuation distribution the integrand targets. A shaper may
also be conditioned on behavior mismatch, \\(\mathrm C(U;B,\hat A)\\), while
still acting through the trainable edge \\(U\\).

On the direct edge, \\(\mathrm M(E)\\) means pure selection;
\\(\mathrm W(E)\\), optionally preceded by a mask, means detached weighting;
and \\(\mathrm C(E)\\), again optionally pre-filtered, means coupled shaping.

Using \\(\mathrm W(E)\mathrm C(E)\\), with or without \\(\mathrm M(E)\\), is
not the default change-of-measure construction. If both \\(\mathrm W\\) and
\\(\mathrm C\\) reduce to one factor of \\(E\\) in their linear limit, their
product reduces to \\(E^2\\); such a design requires a deliberate alternative
interpretation rather than being counted as one correction.

PPO's active gradient region depends on both the ratio and the sign of the
advantage. Selecting the clipped branch removes gradient in one direction;
this differs from truncating a detached importance weight, which changes a
coefficient before differentiation. The explicit coefficient is given in
[Appendix A.5](#appendix-a5). DAPO changes the sign-dependent bounds
([Yu et al., 2025](https://arxiv.org/abs/2503.14476)). GSPO is also a shaper:
it replaces the token ratio with a length-normalized sequence-geometric ratio,
giving \\(\mathrm C_{\mathrm{geo}}(E)\\) in a coupled topology or
\\(\mathrm C_{\mathrm{geo}}(U)\\) after factorization
([Zheng et al., 2025](https://arxiv.org/abs/2507.18071)).

<figure class="triangle-figure">
  <img src="{{ '/assets/policy-triangle-hero.svg' | relative_url }}" alt="Graphical abstract of the policy triangle. The behavior edge B carries past data correction, the proximal vertex requires credit anchored at A q, the update edge U shapes the current update, and bypass is the special case q equals mu so B equals one and U equals E. Representative cards map IS, rejection sampling, IcePop, and KPop.">
  <figcaption>
    Figure 1. Approximating the \\(q\\)-surrogate. The behavior edge carries
    past/data correction, the proximal vertex anchors the required credit
    \\(A^q\\), and the update edge carries differentiable control. Coupled
    bypass sets \\(q=\mu\\), so \\(B=1\\), \\(A^q=A^\mu\\), and \\(U=E\\);
    nonlinear direct operators on \\(E=BU\\) otherwise represent a collapsed,
    generally non-equivalent treatment. Representative cards map methods from
    Section 2.5.
  </figcaption>
</figure>

### 2.2 Direct and factorized topologies

**Coupled bypass.** Set \\(q\equiv\mu\\). The behavior policy becomes the
proximal anchor, so

<div class="equation">
\[
B=1,\qquad U=E=\frac{\pi}{\mu},\qquad A^q=A^\mu.
\]
</div>

The surrogate then becomes the usual behavior-anchored PPO surrogate. This is
the natural synchronous case: no separate \\(\mu\to q\\) data or credit
realignment is needed. Statistical correction and proximal control are
compressed onto \\(E=U\\), avoiding a separate proximal policy at the cost of
losing causal attribution between the two roles. The reduced gradient is
written explicitly in [Appendix A.3](#appendix-a3).

**Collapsed direct treatment.** A method may instead retain a conceptual
\\(q\\) but apply a nonlinear operator directly to \\(E=BU\\). This can be
compared with the factorized surrogate, but it is not generally an equivalent
implementation because

<div class="equation">
\[
\mathcal O(E)=\mathcal O(BU)
\ne
\mathcal O_B(B)\mathcal O_U(U)
\]
</div>

for clipping, masking, and most divergence gates. Direct methods therefore
either instantiate the coupled special case \\(q=\mu\\) or deliberately
collapse behavior mismatch and update drift into a different robust objective.

**Decoupled.** Keep a distinct frozen \\(q\\). The learner can correct rollout
mismatch on \\(B\\), construct or assume a \\(q\\)-anchored credit signal, and
constrain the current update on \\(U\\). A typical implementation has the form
\\(\mathrm M(B)\mathrm W(B)\mathrm C(U;\hat A^{\mathrm{roll}})\\), with an
identity mask when no admission filter is required.

For PPO clipping, the practical difference is attachment. Coupled PPO applies
one nonlinear shaper to the total ratio \\(E\\). Decoupled PPO first attaches a
detached behavior coefficient to \\(B\\), then applies the differentiable PPO
shaper only to \\(U\\), usually while reusing rollout-generated credit. Their
unclipped token coefficients can coincide when the same credit signal is used,
but their clipped regions do not. The side-by-side objectives and the
unclipped identity are in [Appendix A.4](#appendix-a4).

That unclipped token-wise coincidence does **not** establish exactness with respect to
\\(g_q\\): the ideal surrogate uses \\(B_{1:t}A_t^q\\), not merely
\\(B_t\hat A_t^{\mathrm{roll}}\\). Decoupling changes where the nonlinear
trust-region mechanism is attached—coupled PPO clips total mismatch \\(E\\),
whereas Decoupled PPO treats \\(B\\) as behavior correction and clips only
update drift \\(U\\). It also makes the two remaining approximation questions
visible: how much of the prefix ratio to retain and how to align rollout credit
to \\(A^q\\). The choice of \\(q\\) is therefore a speed–stability and
credit-target design variable, not merely an accounting device.

The proximal anchor is itself a design variable. A-3PO, for example,
approximates it by interpolating behavior and current log-probabilities to avoid
an additional model forward pass while retaining the factorization \\(E=BU\\)
([Li et al., 2025](https://arxiv.org/abs/2512.06547)).

### 2.3 Correction geometry and detached tail treatments

This section expands approximations to the past/data factor \\(B_{1:t}\\). The
optimizer-side shaper \\(\mathrm C(U)\\) is held conceptually separate, and the
credit signal is assumed to target \\(A^q\\) unless explicitly marked as
rollout-generated. When the integrand is a raw response return, a sequence
ratio also performs future/credit realignment; this is why ratio horizon cannot
be interpreted without stating the credit anchor.

Token/sequence and IS/TIS/MIS/RS answer different questions. The first pair
chooses **which probability object is measured**; the second chooses **what is
done with its tail**. Treating these as two independent axes makes the
terminology in the analyses of
[Li and Liu, 2025a](https://richardli.xyz/post/rl-collapse-part2/) and
[2025b](https://richardli.xyz/post/rl-collapse-part3/) much easier to compare.

> **In plain language.** First choose the ruler: compare one token, the prefix,
> or the whole answer. Then choose what to do with extreme readings: keep,
> cap, mask, or reject them. “Sequence” and “TIS” are therefore not competing
> labels; they describe different coordinates of one design.

Let the behavior-edge token ratio be
\\(r_t\equiv B_t=q(y_t\mid h_t)/\mu(y_t\mid h_t)\\). In a bypass topology, the
same discussion applies with \\(r_t=E_t\\). Let
\\(\phi_t=\hat A_t^{q\leftarrow\mu}
\nabla_\theta\log\pi_\theta(y_t\mid h_t)\\) denote a local gradient contribution
whose credit has already been aligned to \\(q\\), and \\(F(y)\\) a generic
response-level integrand, such as a raw reward, loss, or full-response gradient
contribution. The symbol \\(F\\) is a placeholder for what is being estimated,
not another policy or correction.

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
| Prefix \\(R_{1:t}\\) | Changes measure through the sampled token at position \\(t\\), without ratios from future tokens. | Exact for a prefix-measurable contribution \\(f_t(h_t,y_t)\\). If its reward or advantage still depends on the sampled suffix, prefix IS alone is generally biased. |
| Sequence \\(R\\) | Changes the complete response measure: \\(R=q(y\mid x)/\mu(y\mid x)\\). | Untruncated sequence IS is exact under support and common-dynamics assumptions, but its second moment and weight concentration can grow rapidly with length. |
| Geometric \\(G\\) | Measures average sampled log-ratio per token. | Length normalized and useful for gating or diagnostics, but **not** a density ratio and therefore not an IS weight. Opposite-signed token log-ratios can also cancel. |

</div>

There is therefore no estimand-independent answer to whether prefix or
sequence correction is “the” unbiased one. For an arbitrary response-level
quantity \\(F(y)\\), the full sequence ratio is the general exact correction.
For a causal contribution \\(f_t\\) that is measurable from \\((h_t,y_t)\\),
future ratios integrate out under the usual assumptions.

In that causal case, Prefix-IS is the minimum-horizon exact correction;
Seq-IS remains unbiased but multiplies by unnecessary future ratios. In common
outcome-level LLM RL, however, \\(\hat A_t\\) is often computed from the
complete response. Then \\(\phi_t\\) depends on the suffix and is not
prefix-measurable: Prefix-IS is not generally exact unless the future return
has already been replaced by the correct target-policy conditional value or
corrected per-decision. The two change-of-measure identities and their
measurability condition are derived in [Appendix A.6](#appendix-a6).

For the \\(q\\)-surrogate, this gives a concrete compatibility rule:

<div class="equation">
\[
\boxed{
B_{1:t}A_t^q
\quad\text{or}\quad
B_{1:t}\!\left(B_{t+1:T}\mathcal R\right)
=B_{1:T}\mathcal R
}
\]
</div>

The first form uses an explicitly aligned advantage and needs only past
correction. The second uses a behavior-generated suffix and lets the future
ratio perform credit realignment. Replacing either expression by
\\(B_t\hat A_t^{\mathrm{roll}}\\) is a lower-variance practical approximation,
not an exact identity.

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
\text{MIS:}\;& \mathbb{I}\!\left[R\le C\right]RF,\\
\text{RS:}\;& \mathbb{I}\!\left[S\in\mathcal A\right]F.
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
Thus Seq-MIS, \\(\mathbb{I}[R\le C]RF\\), is one particular composition; it is
not a synonym for every sequence-level rejection rule. The pure geometric rule
called Geo-Mask in Li and Liu's Part 3, and Geo-RS in veRL, instead has the form
\\(\mathbb{I}[C_{\rm low}\le G\le C_{\rm high}]F\\). It performs selection but
no change of measure. Its geometric \\(k_3\\)-style statistic can estimate a
per-token KL, and the normalization makes the acceptance criterion
length-invariant; the gate nevertheless changes the target by omitting the
rejected region. Geo-RS is therefore best read as length-normalized selection.
The expectation-level statement is in [Appendix A.7](#appendix-a7).

At token scale, replace \\(R,F\\) by \\(r_t,\phi_t\\). The resulting matrix makes
the combinations explicit:

<div class="table-wrap" markdown="1">

| Scale | IS | TIS | MIS | Pure RS |
|---|---|---|---|---|
| Token | \\(r_t\phi_t\\) | \\(\min(r_t,C)\phi_t\\) | \\(\mathbb{I}[r_t\le C]r_t\phi_t\\) | \\(\mathbb{I}[r_t\in\mathcal A_t]\phi_t\\) |
| Sequence | \\(RF\\) | \\(\min(R,C)F\\) | \\(\mathbb{I}[R\le C]RF\\) | \\(\mathbb{I}[R\in\mathcal A]F\\) |
| Geometric gate | Not valid: \\(G\\) is not a density ratio | A heuristic shaper, not TIS in the change-of-measure sense | \\(\mathbb{I}[G\in\mathcal A]RF\\): geometric mask plus sequence IS | \\(\mathbb{I}[G\in\mathcal A]F\\): Geo-RS |

</div>

The matrix gives algebraic descriptions, not universally standardized method
names. In particular, “Token-MIS” and “Seq-RS” may be labeled differently by
different codebases; the formula should take precedence over the acronym.

#### Reading the bias–variance trade-off

The operator determines how the tail is treated; the geometry determines what
kind of mismatch remains:

1. **Seq-IS:** the general unbiased change of measure for an arbitrary
   response-level \\(F(y)\\), but potentially catastrophic weight variance and
   poor effective sample size for long responses.
2. **Prefix-IS:** unbiased for a prefix-measurable causal term and lower
   variance than adding unnecessary suffix ratios. It is generally biased for
   an outcome-level term whose sampled reward or advantage depends on the
   suffix.
3. **Token-IS:** much lower variance under bounded per-token terms, but prefix
   or occupancy bias remains even before any truncation is applied.
4. **TIS:** caps tail influence and usually lowers variance, while adding
   truncation bias. Seq-TIS broadcasts one capped response weight to every
   token; Token-TIS adds truncation bias on top of the token approximation.
5. **MIS:** bounds the accepted IS weight and removes suspected bad-tail data,
   but sacrifices sample efficiency. For Seq-MIS the bias is exactly the
   omitted target-tail contribution.
6. **RS:** can reject on a more robust or length-normalized statistic, but is a
   selection mechanism rather than off-policy correction. Geo-RS avoids a raw
   product threshold's length scale, yet can miss localized token outliers or
   cancellation; it is often paired with Token-TIS to obtain a sequence gate
   plus local weights.

The exact TIS and MIS bias terms are recorded in
[Appendix A.7](#appendix-a7).

<figure class="triangle-figure">
  <img src="{{ '/assets/bias-variance-tradeoff.svg' | relative_url }}" alt="Two-panel bias-variance diagram. The first panel shows that variance exposure typically increases from token to prefix to sequence correction: prefix IS is the minimum exact horizon for a prefix-measurable causal term, while sequence IS is required for an arbitrary response-level term. The second panel shows raw IS retaining the most tail variance with no operator-induced bias, TIS capping the tail, MIS deleting it, and pure rejection sampling as selection rather than change of measure.">
  <figcaption>
    Figure 3. Bias–variance has two independent sources. Ratio horizon controls
    which estimand is exactly corrected and how much weight dispersion is
    accumulated; the tail operator controls how far the estimator departs from
    raw change of measure. Positions are schematic rather than quantitative.
  </figcaption>
</figure>

These rankings require qualifications. Bounds such as polynomial token-level
variance or capped sequence-level variance assume bounded score and
return/advantage terms; stale or incorrect advantages can still dominate. A
fixed threshold can also create length-conditioned truncation or rejection, so
ESS and acceptance rates should always be reported by response length.
Finally, detached TIS is not PPO clipping: TIS caps a statistical weight,
whereas PPO uses an advantage-dependent differentiable shaper as described in
Section 2.1.

A broad class of critic-free policy-loss gradients can now be read as an
explicit composition:

<div class="equation">
\[
\widehat g=
\frac{1}{Z}\sum_i
\mathrm M_i\,\overline{\mathrm W}_i\,\mathrm C_i\,
\hat A_i^{\,\alpha}
\nabla_\theta\log\pi_\theta(y_i\mid h_i),
\qquad
\alpha\in\{\mathrm{roll},\mu,q,\pi\}.
\tag{2}
\]
</div>

A concrete attachment is specified by
\\((\text{edge},\text{operator},\text{scope},\text{statistic},
\text{sign rule},\text{normalization},\text{credit anchor})\\). Exactness is a
property of this full tuple relative to a stated estimand; it is not a property
of a ratio or operator name in isolation.

### 2.4 Combining mitigation mechanisms

The factorization \\(E=BU\\) does not make interventions interchangeable. Where
an operator is attached determines what mismatch it sees, how often it must be
recomputed, and which bias–variance trade-off it introduces.

#### Nonlinear placement matters

<div class="equation">
\[
\operatorname{clip}(E)\ne
\operatorname{clip}(B)\operatorname{clip}(U),\qquad
\mathrm M(E)\ne\mathrm M(B)\mathrm M(U).
\tag{3}
\]
</div>

For \\(k_3(x)=x-1-\log x\\),
the direct statistic also contains a cross-edge interaction term. Therefore,
filtering total mismatch is not equivalent to filtering the behavior and
update edges independently. A mask may nevertheless read any edge without
forcing a downstream importance weight to consume the same edge. The
interaction identity is in [Appendix A.8](#appendix-a8).

#### Credit correction does not always commute with the operators

The credit anchor is a separate design coordinate, but it is not a black-box
module that can always be inserted before or after the other operators. PPO
clipping selects a branch using the sign of \\(\hat A\\); suffix truncation
changes the conditional return being estimated; and a trajectory-dependent
mask can destroy the usual baseline cancellation.

The same issue is stronger for GRPO-style group centering and standardization,
because rejecting one response can change the credit assigned to the others.
Thus a method may classify credit realignment separately, but its masked,
clipped, or group-normalized objective must still be derived jointly. TRM, for
example, notes that its reward-form and advantage-form masked objectives
coincide only when the mask is identically one
([Li et al., 2025](https://arxiv.org/html/2512.23075v5#A7)). The formal
non-cancellation statement is in [Appendix A.8](#appendix-a8).

#### Normalization after truncation

After an IS geometry and truncation rule have been chosen, self-normalization
is an orthogonal variance-control decision. It rescales accepted weights to
keep their average near one. Its domain—token, sequence, prompt group, or
batch—is part of the algorithm. The precise formula is given in
[Appendix A.8](#appendix-a8). It stabilizes mean gradient scale but is biased
at finite sample size
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

### 2.5 Mapping existing methods into the solution space {#taxonomy}

The table maps representative methods by topology and by their dominant
solution-space coordinates. It does not attempt to enumerate all critic,
trace, or group-advantage constructions, but records the credit assumption
because that assumption determines what the displayed ratio can estimate.
Reward shaping, entropy bonuses, dynamic sampling, and systems optimizations
remain outside the table.

<div class="table-wrap" markdown="1">

| Method family | Topology | Solution-space coordinate | Credit assumption | Mitigation role |
|---|---|---|---|---|
| [PPO](https://arxiv.org/abs/1707.06347) / [GRPO](https://arxiv.org/abs/2402.03300) / [DAPO](https://arxiv.org/abs/2503.14476) | Coupled bypass | \\(\mathrm C_{\mathrm{tok}}(E)\\), with \\(q=\mu\\) | Fresh-rollout credit; ideally \\(A^\mu=A^q\\). | One direct ratio carries correction and proximal-control roles; DAPO uses asymmetric bounds. |
| [CISPO](https://arxiv.org/abs/2506.13585) / [TOPR](https://arxiv.org/abs/2503.14286) | Direct | \\(\mathrm W_{\mathrm{tok}}(E)\\) | Rollout credit; no explicit continuation realignment. | Detached clipped or tapered weight; gradients need not vanish outside a PPO band. |
| [GSPO](https://arxiv.org/abs/2507.18071) | Coupled bypass | \\(\mathrm C_{\mathrm{geo}}(E)\\) | Rollout credit under the coupled anchor. | Sequence-geometric shaping; the original objective does not add a separate pre-filter. |
| [Decoupled PPO](https://arxiv.org/abs/2110.00641) / [AReaL](https://arxiv.org/abs/2505.24298) / [A-3PO](https://arxiv.org/abs/2512.06547) | Factorized | \\(\mathrm W(B)\mathrm C(U)\\) | Usually \\(\hat A^{\mathrm{roll}}\approx A^q\\); no explicit \\(\mu\to q\\) credit correction. | Behavior correction is detached from the trainable proximal constraint. |
| [TIS](https://doi.org/10.1198/106186008X320456) / rollout IS | Either | \\(\mathrm W_{\mathrm{tok/prefix/seq}}(B\text{ or }E)\\) | Prefix exactness requires target-policy credit; Seq-IS with raw return can also realign the suffix. | Raw or truncated detached weights trade change-of-measure fidelity for variance control. |
| MIS | Either | \\(\mathrm M_{\mathrm{tok/seq}}(B\text{ or }E)\mathrm W_{\mathrm{raw}}(B\text{ or }E)\\), optional \\(\mathrm C\\) | External credit; selection can alter baseline semantics. | A hard mask removes the tail; accepted samples retain their raw IS weight. |
| RS / Geo-RS | Either | \\(\mathrm M_{\mathrm{tok/seq/geo}}(B\text{ or }E)\\), optional separate \\(\mathrm W\\) and \\(\mathrm C\\) | External credit; no automatic realignment. | Pure selection is orthogonal to weighting; a geometric gate is length normalized but is not an IS ratio. |
| [IcePop](https://arxiv.org/abs/2510.18855) | Factorized | \\(\mathrm M^{\mathrm{ratio}}_{\mathrm{tok}}(B)\mathrm W_{\mathrm{raw,tok}}(B)\mathrm C^{\mathrm{PPO}}_{\mathrm{tok}}(U)\\) | Rollout credit is reused as \\(A^q\\). | A two-sided ratio mask admits a token, the admitted token retains raw behavior correction, and PPO shapes update drift. |
| [KPop](https://ringtech.notion.site/kpop) | Factorized | \\(\mathrm M^{\mathrm{biKL}}_{\mathrm{tok}}(\mu,q)\mathrm W_{\mathrm{raw,tok}}(B)\mathrm C^{\mathrm{PPO}}_{\mathrm{tok}}(U)\\) | Rollout credit is reused as \\(A^q\\). | KPop replaces IcePop's fixed ratio band with bidirectional binary-KL geometry sensitive to absolute token probability. |
| [DPPO](https://arxiv.org/abs/2602.04879) / [TRM](https://arxiv.org/abs/2512.23075) | Direct/collapsed | Divergence \\(\mathrm M(E)\\) or \\(\mathrm C(E)\\) | Rollout/reference credit; no separate proximal-credit correction. | Distributional geometry replaces sampled-ratio magnitude; TRM rejects a sequence on worst-token divergence. |

</div>

The outside-region semantics of IcePop and KPop are masks, not truncation. In
the factorized implementation their behavior-edge coefficients are

<div class="equation">
\[
\omega_t^{\mathrm{IcePop}}
=B_t\mathbb{I}\!\left[\alpha\le B_t\le\beta\right],\qquad
\omega_t^{\mathrm{KPop}}
=B_t\mathbb{I}\!\left[
D_{\mathrm{KL}}^B(q_t\Vert\mu_t)\le\phi,
D_{\mathrm{KL}}^B(\mu_t\Vert q_t)\le\phi
\right],
\]
</div>

where \\(D_{\mathrm{KL}}^B\\) is binary KL on “the sampled token” versus “all
other tokens.” Outside either admission region, \\(\omega_t=0\\): the token
contributes no policy gradient. Inside, the raw \\(B_t\\) weight remains, and
PPO separately shapes \\(U_t\\). Truncation would instead replace an outlying
\\(B_t\\) by a boundary value and keep updating it. This mask–weight
composition is explicit in the IcePop objective and in the
[AReaL IcePop/KPop integration](https://github.com/areal-project/AReaL/pull/1405).
The sources sometimes use “clipping ratio” informally for the fraction of
filtered tokens; that reporting term does not change the zero-policy-gradient
mask defined by the loss.

#### What the mapping clarifies

1. **Coupled bypass is the degenerate anchor \\(q=\mu\\).** Then \\(B=1\\),
   \\(U=E\\), and fresh-rollout credit targets both \\(A^\mu\\) and \\(A^q\\).
   A direct operator on \\(E=BU\\) with a distinct latent \\(q\\) is instead a
   collapsed, generally non-equivalent treatment.
2. **PPO and GSPO clipping are not pre-filters.** Their branches change gradient
   flow in an advantage-dependent direction; a detached mask rejects
   independently of that direction.
3. **A decoupled filter may read any raw edge.** \\(\mathrm M(B)\\) is static when
   \\(q\\) is frozen, whereas \\(\mathrm M(U)\\) and \\(\mathrm M(E)\\) generally
   change as \\(\pi\\) is updated.
4. **IcePop and KPop are masked IS, not TIS.** Both reject an out-of-region
   token and retain \\(B\\) for an accepted token; KPop changes the admission
   geometry from a fixed ratio band to bidirectional binary KL.
5. **A correct ratio does not imply correct credit.** Prefix correction is exact
   for the \\(q\\)-surrogate only when paired with \\(A^q\\); the mapped
   factorized methods generally reuse rollout credit instead of explicitly
   realigning the suffix.

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
works analyze parts of the credit problem more deeply than the anchor label
used here. The triangle records whether credit targets \\(\mu\\), \\(q\\), or
\\(\pi\\), but does not attempt to subsume the full theory of critic learning,
trace estimators, or group-normalized advantages.
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
| Rollout continuation differs from the proximal policy | Credit anchor \\(\hat A^{q\leftarrow\mu}\\) | Re-estimate \\(A^q\\) with suffix correction, a \\(q\\)-critic, traces, or re-rolls instead of assuming rollout credit is unchanged. |
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

1. **Explicit proximal-credit realignment:**
   \\(\mathrm W_{\mathrm{prefix}}(B)
   \mathcal R_A^{\mu\to q}\mathrm C(U)\\), where
   \\(\mathcal R_A^{\mu\to q}\\) uses a \\(q\\)-critic, a truncated trace, a
   doubly robust estimator, or re-rolled suffixes. Existing mapped methods
   largely leave this module implicit.
2. **Decoupled sequence shaping:**
   \\(\mathrm W_{\mathrm{tok}}(B)\mathrm C_{\mathrm{geo}}(U)\\). Correct rollout
   mismatch token-wise, then constrain the update with a sequence-geometric
   proximal ratio.
3. **Heterogeneous two-edge mitigation:** apply entropy-adaptive admission and
   \\(\mathrm W(B)\\) on the behavior edge, followed by a divergence-based mask
   or shaper on \\(U\\). Each edge is treated with geometry appropriate to its
   source.
4. **Segment-aware asynchronous mitigation:** normalize or truncate
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
\tag{4}
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
3. **Identify the credit anchor.** State whether the loss uses rollout credit,
   estimates \\(A^q\\), or targets the exact current-policy advantage.
4. **Locate every intervention.** Record its edge, operator, statistic,
   geometry, sign rule, and normalization domain.
5. **Inspect compositions.** Distinguish deliberate bias–variance choices from
   accidental missing or duplicated ratio factors, and check whether masking
   or group normalization changes the advantage semantics.
6. **Measure the coordinates separately.** Report token \\(\log B_t\\),
   cumulative prefix and sequence log-ratios, \\(U\\), and \\(E\\). Track
   effective sample size before and after truncation, truncation or rejection
   rates by response length, disagreement between rollout and proximal credit,
   gradient variance, policy age, reward, and throughput.

## 4. Limitations and conclusion {#scope}

The policy triangle organizes the mitigation solution space; it does not rank
its coordinates. It cannot choose thresholds, guarantee that masking improves
return, repair support mismatch, or replace empirical validation. It identifies
credit realignment as a missing module but does not supply or compare a
general-purpose \\(A^{q\leftarrow\mu}\\) estimator; critic learning, trace
methods, and finite-group advantage theory remain open. It also omits
reward-model drift, environment nonstationarity, and optimizer-state staleness.
Sequence-product importance sampling can be formally exact and statistically
unusable, while geometric statistics can be stable without being exact
change-of-measure factors.

Within this scope, the behavior, proximal, and current policies provide a
compact set of anchors. Masks select regions, detached weights change measure,
and differentiable shapers control the update; each chooses its own geometry
and must declare the policy under which credit is defined. Coupled bypass is
the degenerate case \\(q=\mu\\), while a direct operator on \\(E=BU\\) with a
distinct latent \\(q\\) is a collapsed, generally non-equivalent treatment.
Decoupling separates exogenous rollout mismatch, proximal credit, and
endogenous update drift.

The main value is a clearer view of the whole solution space. The
edge–operator–geometry representation separates **where mismatch is measured**,
**what a mitigation mechanism does**, and **at what scale it acts**; the credit
anchor states **which continuation policy the update evaluates**. Existing
methods become recognizable approximations of a shared \\(q\\)-surrogate rather
than a list of unrelated losses. The same coordinates also expose open regions
that can be tested without presenting them as established improvements.

### Suggested citation {#citation}

    @misc{zhao2026policytriangle,
      title  = {The Policy Triangle: A Unified View of Policy-Mismatch
                Mitigation in LLM Reinforcement Learning},
      author = {Huaiyi Zhao},
      year   = {2026},
      url    = {https://huaiyizhao.github.io/policy-triangle/}
    }

## Technical appendix: derivations and exactness conditions {#appendix}

This appendix contains the algebra behind the conclusion formulas in the main
text. It is intentionally skippable: the policy-triangle framework can be read
without it.

### A.1 Exact change of measure {#appendix-a1}

Define \\(s_t=\nabla_\theta\log\pi_\theta(y_t\mid h_t)\\), together with

<div class="equation">
\[
E_i=\frac{\pi(y_i\mid h_i)}{\mu(y_i\mid h_i)},
\qquad
E_{a:b}=\prod_{i=a}^{b}E_i.
\tag{A.1}
\]
</div>

Under support and common-dynamics assumptions, changing the prefix measure from
\\(\mu\\) to \\(\pi\\) gives

<div class="equation">
\[
\nabla J(\pi)
=
\sum_t\mathbb E_\mu
\left[
E_{1:t}A_t^\pi(h_t,y_t)
\nabla_\theta\log\pi_\theta(y_t\mid h_t)
\right].
\tag{A.2}
\]
</div>

If the integrand uses a sampled terminal return rather than the
current-policy conditional advantage, the future measure must also be changed.
Writing the terminal return as \\(\mathcal R(y)\\),

<div class="equation">
\[
\nabla J(\pi)
=
\sum_t\mathbb E_\mu
\left[
E_{1:T}\mathcal R(y)
\nabla_\theta\log\pi_\theta(y_t\mid h_t)
\right].
\tag{A.3}
\]
</div>

Equation (A.2) exposes past occupancy correction and current-policy credit
separately; Equation (A.3) lets the full sequence ratio perform both jobs.

### A.2 From the performance-difference identity to TRPO {#appendix-a2}

For a frozen reference policy \\(q\\), the performance-difference identity is

<div class="equation">
\[
J(\pi)-J(q)
=
\sum_t
\mathbb E_{h_t\sim d_t^\pi,\,y_t\sim\pi}
\left[A_t^q(h_t,y_t)\right].
\tag{A.4}
\]
</div>

Replacing the current occupancy \\(d_t^\pi\\) with the frozen occupancy
\\(d_t^q\\) yields the TRPO local surrogate shown in the main text. The
replacement is locally consistent because

<div class="equation">
\[
L_q^{\mathrm{TRPO}}(q)=J(q),
\qquad
\left.\nabla L_q^{\mathrm{TRPO}}(\pi)\right|_{\pi=q}
=
\left.\nabla J(\pi)\right|_{\pi=q}.
\tag{A.5}
\]
</div>

The equality is first-order and local; it does not imply equal gradients after
\\(\pi\\) has moved far from \\(q\\).

### A.3 The three-policy factorization {#appendix-a3}

At each sampled token,

<div class="equation">
\[
\frac{\pi_t}{\mu_t}
=
\underbrace{\frac{q_t}{\mu_t}}_{B_t}
\cdot
\underbrace{\frac{\pi_t}{q_t}}_{U_t}.
\tag{A.6}
\]
</div>

Changing the history and sampled-action measure from \\(\mu\\) to \\(q\\)
therefore gives

<div class="equation">
\[
\mathbb E_{h_t\sim d_t^q,\,y_t\sim q}
\left[
U_tA_t^q s_t
\right]
=
\mathbb E_{y\sim\mu}
\left[
B_{1:t}U_tA_t^q s_t
\right].
\tag{A.7}
\]
</div>

If only a return from the behavior-generated suffix is observed, the missing
\\(q\\)-continuation can be expressed as

<div class="equation">
\[
Q_t^q(h_t,y_t)
=
\mathbb E_\mu
\left[
B_{t+1:T}\mathcal R
\mid h_t,y_t
\right],
\qquad
g_q^{\mathrm{reward}}(\pi)
=
\sum_t\mathbb E_\mu
\left[
B_{1:T}U_t\mathcal R\,s_t
\right].
\tag{A.8}
\]
</div>

In coupled bypass, \\(q=\mu\\), so \\(B=1\\), \\(U=E\\), and the same expression
reduces to

<div class="equation">
\[
g_\mu(\pi)
=
\sum_t\mathbb E_\mu
\left[E_tA_t^\mu s_t\right].
\tag{A.9}
\]
</div>

### A.4 Coupled and decoupled PPO objectives {#appendix-a4}

Coupled PPO attaches its nonlinear shaper to the direct ratio:

<div class="equation">
\[
L_{\mathrm{coupled}}
=
\mathbb E_\mu
\left[
\min\!\left(
E_t\hat A_t^\mu,\,
\operatorname{clip}(E_t,1-\epsilon,1+\epsilon)\hat A_t^\mu
\right)
\right].
\tag{A.10}
\]
</div>

Practical token-level Decoupled PPO instead separates a detached behavior
coefficient and a trainable update shaper:

<div class="equation">
\[
L_{\mathrm{decoupled}}^{\mathrm{tok}}
=
\mathbb E_\mu
\left[
\operatorname{sg}[B_t]\,
\min\!\left(
U_t\hat A_t^{\mathrm{roll}},\,
\operatorname{clip}(U_t,1-\epsilon,1+\epsilon)
\hat A_t^{\mathrm{roll}}
\right)
\right].
\tag{A.11}
\]
</div>

When clipping is inactive and the same credit signal is used,

<div class="equation">
\[
\operatorname{sg}[B_t]U_t\hat A_t^{\mathrm{roll}}
\nabla_\theta\log\pi_\theta
=
E_t\hat A_t^{\mathrm{roll}}\nabla_\theta\log\pi_\theta.
\tag{A.12}
\]
</div>

This token identity does not recover the ideal prefix factor
\\(B_{1:t}A_t^q\\); it only explains why the two unclipped token coefficients
can look identical.

### A.5 PPO's advantage-dependent active region {#appendix-a5}

Ignoring boundary points, the effective PPO token coefficient can be written

<div class="equation">
\[
\mathrm C^{\mathrm{PPO}}(X,\hat A)=
X\,\mathbb I\!\left[
(\hat A\ge0 \land X\le1+\epsilon_+)
\;\lor\;
(\hat A&lt;0 \land X\ge1-\epsilon_-)
\right].
\tag{A.13}
\]
</div>

The indicator describes whether gradient flows through the unclipped branch.
Because the condition depends on the sign of \\(\hat A\\), PPO clipping is a
differentiable, direction-dependent shaper rather than detached TIS.

### A.6 When prefix and sequence IS are exact {#appendix-a6}

For an arbitrary response-level integrand \\(F(y)\\), raw sequence IS satisfies

<div class="equation">
\[
\mathbb E_{y\sim\mu}[R(y)F(y)]
=
\mathbb E_{y\sim q}[F(y)],
\qquad
R=\prod_{t=1}^T\frac{q_t}{\mu_t}.
\tag{A.14}
\]
</div>

If \\(f_t\\) is measurable from \\((h_t,y_t)\\), future ratios integrate to one:

<div class="equation">
\[
\mathbb E_\mu\!\left[Rf_t(h_t,y_t)\right]
=
\mathbb E_\mu\!\left[R_{1:t}f_t(h_t,y_t)\right]
=
\mathbb E_q\!\left[f_t(h_t,y_t)\right].
\tag{A.15}
\]
</div>

When a sampled advantage depends on the suffix, it is not such an \\(f_t\\).
Exactness can instead be recovered by explicitly aligning the credit:

<div class="equation">
\[
B_{1:t}A_t^q
\qquad\text{or}\qquad
B_{1:t}\!\left(B_{t+1:T}\mathcal R\right)=B_{1:T}\mathcal R.
\tag{A.16}
\]
</div>

### A.7 Tail operators and their bias {#appendix-a7}

For TIS with upper cap \\(C\\), the operator-induced bias relative to raw
sequence IS is

<div class="equation">
\[
\operatorname{Bias}_{\mathrm{TIS}}
=
\mathbb E_\mu\!\left[(\min(R,C)-R)F\right].
\tag{A.17}
\]
</div>

For one-sided Seq-MIS,

<div class="equation">
\[
\operatorname{Bias}_{\mathrm{MIS}}
=
-\mathbb E_q\!\left[F\,\mathbb I(R>C)\right].
\tag{A.18}
\]
</div>

Let a geometric gate accept when \\(G\in\mathcal A\\). Pure Geo-RS is a
selected-data objective rather than a density-ratio estimator. Even after
adding valid sequence IS,

<div class="equation">
\[
\mathbb E_\mu\!\left[\mathbb I(G\in\mathcal A)RF\right]
=
\mathbb E_q\!\left[\mathbb I(G\in\mathcal A)F\right].
\tag{A.19}
\]
</div>

Its bias relative to the unmasked target is therefore

<div class="equation">
\[
-\mathbb E_q\!\left[
F\,\mathbb I(G\notin\mathcal A)
\right].
\tag{A.20}
\]
</div>

Length normalization changes the geometry of the gate; it does not remove this
omitted-region term.

### A.8 Composition, baselines, and normalization {#appendix-a8}

Nonlinear edge placement introduces interactions. For the sampled
\\(k_3(x)=x-1-\log x\\) statistic,

<div class="equation">
\[
k_3(BU)=k_3(B)+k_3(U)+(B-1)(U-1).
\tag{A.21}
\]
</div>

A trajectory-dependent mask can also invalidate ordinary baseline
cancellation:

<div class="equation">
\[
\mathbb E_\mu
\left[
M(y)B_{1:t}U_t\,b(h_t)s_t
\right]
\ne0
\qquad\text{in general}.
\tag{A.22}
\]
</div>

Finally, after masking or truncation, one common self-normalization is

<div class="equation">
\[
\overline w_i=
\frac{\widetilde w_i}
{\sum_jM_j\widetilde w_j/\sum_jM_j}.
\tag{A.23}
\]
</div>

It stabilizes average gradient scale but changes the finite-sample estimand.

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
