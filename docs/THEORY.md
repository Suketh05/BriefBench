# A Depth Theory of Retrieval Failure — Derivation Walkthrough

This document derives, step by step, the theory behind *Depth, Not Length*: why
similarity retrieval collapses super-geometrically as a governing decision moves
further from the task surface, why a typed store that follows stored links pays
only a steady per-hop cost, and where the two regimes cross over. Every closed
form below is implemented in [`src/membench/theory/`](../src/membench/theory) and
exposed through the `membench theory` command. The walkthrough is grounded in
those five modules — [`decay.py`](../src/membench/theory/decay.py),
[`recovery.py`](../src/membench/theory/recovery.py),
[`crossover.py`](../src/membench/theory/crossover.py),
[`information.py`](../src/membench/theory/information.py), and
[`fit.py`](../src/membench/theory/fit.py) — and quotes the governing equations
verbatim from the paper.

**Provenance discipline.** Equations are quoted verbatim from the paper LaTeX via
the verified paper digest. Numbers carry their tier: *paper-reported* figures come
from the paper; *repo-measured* figures come from committed result files under
[`results/`](../results) and are labelled as such. The two are never presented as
identical.

---

## 0. The object of study: decision-compliance

At each edit a coding agent must recover the product knowledge — decisions,
constraints, rationale — that silently governs the code it touches, and must honor
a decision once it supersedes an earlier one. The paper calls this
*decision-compliance* and factors it into two independent stages (paper-reported,
`eq:factor`):

> P_comply(d) = P_ret(d) · κ(d)

where `P_ret(d)` is the probability the governing decision is *retrieved* at causal
depth `d`, and `κ(d)` is the **use factor** — `P(comply | n* retrieved)`, the
conditional probability the agent actually *acts* on a decision once it is in
context. The whole theory that follows is a model of `P_ret(d)`; the use factor
`κ` is treated separately in [§5](#5-the-compliance-factorization-and-the-κ-ceiling).

A task is *governed* when the correct action is not a function of the task surface
alone — formally (paper-reported, `def:governed`) governed iff `H(Y|X)>0` iff
`I(Y;D|X)>0`, with `Y=φ(X,D)`. For such tasks any context-free agent has a
non-zero error floor: the tightest bound is the Bayes error (paper-reported,
`eq:irreducible`, `thm:irreducible`):

> P_err^cf ≥ P_e* := 1 − E_X max_y P(y|X) > 0

This is why context is *necessary*, not merely helpful — and it is consistent with
the `none` arm scoring compliance **0.0** on synthetic at depth 1, 2, and 3
(repo-measured, [`results/METRICS.md`](../results/METRICS.md),
`compliance_synthetic_d{1,2,3}` rows, `claude/synthetic`).

---

## 1. The per-hop survival model and the geometric decay law

Acting correctly on a task at the head of a dependence chain
`n_0 → n_1 → … → n_d` requires recovering *every* upstream link. Under hop
independence (paper-reported, `ass:hopindep`), full-chain retrieval is the product
of per-hop survival probabilities (paper-reported, `eq:perhop`):

> P_ret(d) = Π_{k=1}^d f_k

The single empirical input is how *resemblance* to the current query falls with
causal distance. The paper assumes geometric decay with constant retention
(paper-reported, `ass:geometric`): `s_k = s_0·ρ^k`, with `0<ρ<1` and `0<s_0≤1`,
and identifies the per-hop survival with the per-hop similarity, `f_k = s_k`.

This is exactly [`SimilarityDecayModel`](../src/membench/theory/decay.py): `s_i =
s_0 ρ^i`, with `s0 ∈ (0,1]` the retrievability of the node closest to the query
and `ρ ∈ (0,1)` the per-hop decay rate. The module also fits `(s_0, ρ)` from
measured `(distance, similarity)` pairs by ordinary least squares in log-space,
because the geometric law linearises:

```
log s_i = log s_0 + i·log ρ
```

`fit_similarity_decay` returns the slope `ρ = e^slope`, the intercept
`s_0 = e^intercept`, and an `R²` that quantifies how well the geometric assumption
actually holds on the data — the honest check that the whole edifice rests on.

---

## 2. The similarity recovery ceiling and its collapsing slope

Multiplying the per-hop retrievabilities of the decay model over the `d` hops
gives the headline ceiling. Verbatim (paper-reported, `eq:simceiling`,
`thm:sim`):

> P_ret^sim(d) = Π s_0 ρ^k = s_0^d · ρ^{Σk} = **s_0^d · ρ^{d(d+1)/2}**
>
> log P_ret^sim(d) = d·log s_0 + (d(d+1)/2)·log ρ — concave, slope → −∞ (super-geometric).

The exponent `d(d+1)/2` is the whole story. Because it is *quadratic* in depth,
`log P_ret^sim` is concave with slope diverging to `−∞`: chain recovery falls off
faster than any fixed exponential rate. For deep chains the agent is reduced to
guessing the very links that hold the task together.

This is implemented directly in [`recovery.py`](../src/membench/theory/recovery.py):

- `similarity_recovery(d, model)` evaluates `s0**d * rho**(d*(d+1)/2)`.
- `log_similarity_recovery(d, model)` evaluates `d*log(s0) + (d*(d+1)/2)*log(rho)`
  — the log-space form is not cosmetic: the similarity product underflows float64
  for even moderately deep chains, so the crossover solver and plots must operate
  in logs to stay numerically honest.

### A decoder-independent floor

The ceiling above assumes the survivors are recovered at all. The paper also
proves a floor that holds for *any* decoder, via Fano and data processing
(paper-reported, `eq:fano`, `thm:fano`):

> P_err(d) ≥ 1 − (I(n_d;n̂)+1)/log2 M_d ≥ 1 − (I(n_d;S_d)+1)/log2 M_d

As mutual information `I(n_d;S_d) → 0` (the survivor set becomes near-uniform), the
floor approaches `1 − 1/log2 M_d`, with `M_d = Θ(d)`. No cleverness in the
retriever escapes it.

---

## 3. Structured recovery: the bounded-traversal alternative

A typed store does not search by resemblance; it follows an explicit link, which
succeeds with probability `q` close to one independent of whether two connected
facts *look* alike. Under per-edge independence (paper-reported, `ass:edgeindep`),
full-chain structured recovery is (paper-reported, `thm:struct`):

> P_ret^struct(d) = **q^d**, q∈(0,1] per-edge fidelity, Θ(d) work, no M_d penalty.

This is [`structured_recovery(d, q)`](../src/membench/theory/recovery.py), returning
`q**d` for `q ∈ (0,1]`. The contrast with §2 is the entire thesis: a *linear*
exponent (`q^d`) versus a *quadratic* one (`ρ^{d(d+1)/2}`).

The ratio of the two laws,
`P_struct/P_sim = q^d / (s_0^d ρ^{d(d+1)/2}) → ∞` as `d → ∞`, so the advantage of
structure grows without bound with depth. [`RecoveryModel`](../src/membench/theory/recovery.py)
bundles the decay law and `q` and exposes `log_advantage(d) = d·log q −
log P_sim(d)` (computed in log-space precisely so it stays finite where `p_sim`
underflows) and `advantage_ratio(d)`.

### Hitting time and return-on-tokens

The same parameters drive a second, economic separation. The expected work to
recover a depth-`d` chain is (paper-reported, `thm:hitting`):

> T_struct = Θ(d); T_sim = Ω(1/P_ret^sim(d)) = Ω(ρ^{−d(d+1)/2})

[`information.py`](../src/membench/theory/information.py) turns this into the
cost-to-correct model. To capture the node at distance `i`, similarity must surface
`1/s_i` candidates on average, so it injects `T_sim(d) = c·Σ 1/s_i = (c/s_0)·Σ ρ^{−i}`
tokens (geometric in depth), whereas the structured walk injects exactly the `d`
nodes it followed, `T_struct(d) = c·d`, at precision one. The payoff metric is
expected cost-to-correct, `C(d) = T(d)/P_ret(d)`; both numerator and denominator
move against similarity as depth grows, so the cost-to-correct ratio diverges —
the structured store is past the crossover both *more often correct* and *reading
fewer tokens*. The reciprocal, **Return on Tokens** (`P_ret/T`), is the
economic-buyer lens the paper reports.

---

## 4. The gap g(d) and the crossover depth d\*

On pure recovery the structured store always looks better, so to be honest it must
be charged its overhead. Let `V` be the value of solving a task correctly and `W`
the extra per-chain cost of building and maintaining typed links. Structured
memory is the better choice when its net recovery gain clears that overhead. The
governing inequality and gap (paper-reported, `crossover.py` module docstring /
`eq:gd`):

> V·(q^d − s_0^d·ρ^{d(d+1)/2}) > W
> ⟺ g(d) := P_struct(d) − P_sim(d) > τ, with τ = W/V

so the **structured advantage / crossover gap** is (paper-reported, `eq:gd`):

> g(d) = P_ret^struct(d) − P_ret^sim(d) = q^d − s_0^d·ρ^{d(d+1)/2}. Unimodal.

[`crossover.py`](../src/membench/theory/crossover.py) implements this faithfully —
and corrects an over-claim in the theory sketch. The docstring is explicit:

> Proposition 2 asserts `g` is increasing, which holds only in the `q = 1` limit
> (no structured decay). For `q < 1` both terms vanish as `d → ∞`, so `g` rises,
> peaks, then falls back toward zero — the win is an *interval*
> `[d*, d_upper]`, not a half-line.

So `find_crossover_depth(model, tau)` does not assert a half-line. It scans integer
depths `1..max_depth`, finds the contiguous region where `g(d) > τ`, takes its
smallest integer as `d*`, reports the full `win_region`, and refines a continuous
crossover on the rising edge with Brent's method (`d_star_continuous`). `τ` itself
is `overhead_ratio(maintenance_cost, task_value) = W/V`.

### The calibrated operating point

The paper calibrates the synthetic operating point to `s_0=0.70, ρ=0.67, q=0.97`
and reports (paper-reported, `prop:cross`):

> With s_0=0.70, ρ=0.67, q=0.97: g(1)≈0.50, g(2)≈0.79. Crossover d*=1 for τ≤0.50;
> **d*=2 for 0.50<τ≤0.79**. Closed-form crossover (smallest d with g>0) = d*=1.
> Reported d*=2 is calibrated post-hoc.

At this operating point, at `d=3` traversal recovers `q^3 ≈ 0.91` while similarity
recovers `s_0^3 ρ^6 ≈ 0.031` — a ~29× gap (paper-reported, in-sample fit). The
honesty flag matters: the *closed-form* crossover (smallest `d` with any positive
gain) is `d*=1`; the reported `d*=2` is the calibrated value that emerges once the
typed store is charged an overhead `τ` in the band `(0.50, 0.79]`. The paper marks
this as calibrated post-hoc, not proven.

---

### The `membench theory` command

The crossover is exposed as a CLI command
([`src/membench/cli.py`](../src/membench/cli.py), `theory`):

```
uv run membench theory \
    --rho 0.5 \          # ρ, per-hop similarity decay rate, in (0,1)
    --q 0.97 \           # q, structured per-edge fidelity, in (0,1]
    --s0 0.9 \           # s0, retrievability of the nearest node, in (0,1]
    --maintenance 0.2 \  # W, per-chain link maintenance cost (>= 0)
    --value 1.0          # V, value of a correct action (> 0)
```

It builds a `RecoveryModel(SimilarityDecayModel(s0, rho), q)`, computes
`τ = overhead_ratio(maintenance, value) = W/V`, and prints
`find_crossover_depth(model, τ)`.

### Worked example: `uv run membench theory --rho 0.5 --q 0.97`

Running the example exactly as documented in [`README.md`](../README.md) (defaults
`s0=0.9`, `maintenance=0.2`, `value=1.0`) produces:

```
predicted d* = 1 (continuous 1.00); win region (1, 52); tau=0.200
```

**Why `d*=1`.** The overhead ratio is `τ = W/V = 0.2/1.0 = 0.200`. The gain at
depth 1 is `g(1) = q − s_0·ρ = 0.97 − 0.9·0.5 = 0.97 − 0.45 = 0.52`, which already
exceeds `τ = 0.20`. Structure clears its overhead immediately, so the smallest
integer depth in the win region — and hence `d*` — is **1**; the continuous root on
the rising edge coincides at `1.00`. The win is an interval, not a half-line:
`g(d)` keeps exceeding `0.20` up to depth **52**, beyond which `q^d = 0.97^d` has
itself decayed enough that the gap drops below `τ`, closing the region at
`win region (1, 52)`.

To reproduce the paper's calibrated `d*=2`, move to its operating point and charge
the typed store an overhead in the `(0.50, 0.79]` band — e.g.:

```
uv run membench theory --rho 0.67 --q 0.97 --s0 0.70 --maintenance 0.6 --value 1.0
# predicted d* = 2 (continuous 1.26); win region (2, 16); tau=0.600
```

Here `g(1) ≈ 0.50 < τ = 0.60 < g(2) ≈ 0.79`, so the smallest integer depth whose
gain clears the overhead is `d*=2`, exactly the calibrated value of `prop:cross`.

---

## 5. The compliance factorization and the κ ceiling

Recovery is necessary but not sufficient: an agent can have the governing decision
in context and still fail to act on it. That second stage is the use factor `κ`
from `eq:factor` ([§0](#0-the-object-of-study-decision-compliance)). On real code
it does not approach one. The paper proves it acts as a hard ceiling
(paper-reported, `cor:atten`):

> ∂P_comply/∂P_ret = κ; under separability P_comply ≤ κ (hard ceiling).

and gives the real-code ceiling explicitly (paper-reported, `eq:realceiling`):

> P_comply^real(d) = P_ret(d)·κ ≤ κ; κ_sim≈0.6, κ_typed≈0.70.

The empirical consequence is the paper's central real-code finding. On the pooled
real-code datasets (dcbench + swebench, Claude), the typed decision-graph store
(Brief) is the **only** arm whose use factor clears the band that ceilings every
similarity retriever (paper-reported, `tab:realret`, abstract):

| arm | recall P_ret | compliance P_comply | use factor κ | tier |
|---|---|---|---|---|
| Brief (typed graph) | 0.667 | 0.469 | **0.703** | paper-reported |
| dense | 0.635 | 0.406 | 0.639 | paper-reported |
| hybrid_rrf | 0.615 | 0.396 | 0.644 | paper-reported |
| tfidf | 0.604 | 0.385 | 0.637 | paper-reported |
| bm25 | 0.604 | 0.344 | 0.570 | paper-reported |
| rerank_ce | 0.583 | 0.354 | 0.607 | paper-reported |
| raptor | 0.573 | 0.323 | 0.564 | paper-reported |

The similarity arms sit in the **0.56–0.64** band; the typed store's `κ=0.703` is
the only value to clear it (paper-reported). On synthetic, by contrast, the use
factor is `κ≈0.95–0.99` (paper-reported) — the ceiling is a property of *real*
code, where deciding what an in-context decision *implies* is itself hard.

> Separately and not to be conflated: the committed results file reports Brief
> `compliance@all = 0.7037` (repo-measured, [`results/METRICS.md`](../results/METRICS.md),
> `claude/all/d=all`). This is all-data compliance pooled across every dataset
> including synthetic — a different quantity from the paper-reported pooled
> real-code use factor `κ=0.703`. They are numerically close but provenance- and
> definition-distinct; the digest flags this conflation risk directly.

---

## 6. Supersession: honoring the current decision

Decision-compliance includes honoring a decision *once it supersedes an earlier
one* — a query must rank the **current** decision first, not a stale superseded
one. Because a typed store carries an explicit `supersedes` edge, supersession is
dereferenceable; similarity retrieval, which cannot tell "current" from
"superseded but still resembling the query," lands near chance.

On the supersession probe (n=40 paired queries, chance = 50%), the
current-ranked-first rate is (paper-reported, `tab:super`):

| arm | current-ranked-first % | tier |
|---|---|---|
| Brief (typed graph) | 92.3 | paper-reported |
| dense | 68.7 | paper-reported |
| tfidf | 65.8 | paper-reported |
| bm25 | 64.1 | paper-reported |

The similarity arms sit in a **64–69%** band; the typed store leads by **23–28
points** (paper-reported). This is the same mechanism as the depth result, applied
to time rather than distance: the governance edge makes the right answer
dereferenceable instead of a resemblance gamble.

---

## 7. From measurement back to prediction

[`fit.py`](../src/membench/theory/fit.py) closes the loop. The decay law supplies
`(s_0, ρ)` from embedding similarities ([§1](#1-the-per-hop-survival-model-and-the-geometric-decay-law));
the remaining free parameter `q` is estimated from the structured arm's observed
full-chain recovery rates. Because `P_struct(d) = q^d` has no intercept (`P=1` at
`d=0`), `q` is fit by OLS **through the origin** on `(d, log P)`:

```
log q̂ = (Σ_d d·log P_d) / (Σ_d d²),   q̂ = e^{log q̂}
```

`estimate_link_reliability` returns `q` (clipped to `(0,1]`) with a through-origin
`R²`; `build_recovery_model` assembles the `RecoveryModel`; `predict_crossover`
runs the solver of [§4](#4-the-gap-gd-and-the-crossover-depth-d); and
`compare_crossover` checks the predicted `d*` against the empirically measured
changepoint within a tolerance. The headline empirical claim of the paper is then a
single sentence: the theory, parameterised only by measured `ρ, s_0, q`, predicts a
crossover at `d*`, and the accuracy/cost curves are observed to cross there.

---

## Summary of the closed forms

| quantity | closed form | module | tier |
|---|---|---|---|
| per-hop similarity | `s_i = s_0·ρ^i` | [`decay.py`](../src/membench/theory/decay.py) | paper-reported `ass:geometric` |
| similarity recovery ceiling | `P_ret^sim(d) = s_0^d·ρ^{d(d+1)/2}` | [`recovery.py`](../src/membench/theory/recovery.py) | paper-reported `eq:simceiling` |
| structured recovery | `P_ret^struct(d) = q^d` | [`recovery.py`](../src/membench/theory/recovery.py) | paper-reported `thm:struct` |
| crossover gap | `g(d) = q^d − s_0^d·ρ^{d(d+1)/2}` | [`crossover.py`](../src/membench/theory/crossover.py) | paper-reported `eq:gd` |
| overhead ratio | `τ = W/V` | [`crossover.py`](../src/membench/theory/crossover.py) | paper-reported |
| compliance factorization | `P_comply(d) = P_ret(d)·κ(d)` | (theory) | paper-reported `eq:factor` |
| real-code ceiling | `P_comply^real(d) = P_ret(d)·κ ≤ κ` | (theory) | paper-reported `eq:realceiling` |
| token cost (similarity) | `T_sim(d) = (c/s_0)·Σ ρ^{−i}` | [`information.py`](../src/membench/theory/information.py) | paper-reported |
| link-reliability fit | `q̂ = exp(Σ d·log P_d / Σ d²)` | [`fit.py`](../src/membench/theory/fit.py) | repo implementation |

**Integrity note.** No number in this document was invented, estimated, or
recomputed. Paper-reported figures are quoted from the verified paper digest
(cross-checked against the paper LaTeX); the one repo-measured figure (Brief
`compliance@all = 0.7037`) is cited to [`results/METRICS.md`](../results/METRICS.md)
and kept explicitly distinct from the paper-reported `κ=0.703`. The two CLI
transcripts were produced by running `membench theory` at the stated parameters.
