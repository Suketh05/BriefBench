# Figures index

**Provenance.** These are the manuscript's figures, copied verbatim from the paper *Depth, not Length* (`depth_not_length_FIXED.tex`). The paper is the single source of truth; the files in [`figures/`](figures) are exactly the figures the manuscript ships. Captions below are transcribed verbatim from the paper's `\caption{...}` (LaTeX formatting and cross-references trimmed for Markdown). These are the paper's figures, not plots re-measured by this repository.

The paper has **66 figure environments** (**80** unique `\includegraphics`), and the repository ships **108** figure files in `figures/`. Figure numbers below follow the manuscript's float order. Some figure environments combine multiple panels; every referenced panel is listed under its figure.

## Datasets & metric distributions

### Figure 1 — `F098_difficulty_close.png`

Label: `fig:difficulty` · Paper section: Datasets

Dataset difficulty. Mean compliance across arms by dataset: synthetic is separable while dcbench/swebench compress the arms together, foreshadowing the real-code retrieval parity of Section (sec:resreal).

### Figure 2 — `F054_violin_compliance.png`, `F055_ecdf_compliance.png`

Label: `fig:violin_compliance` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Compliance distribution. Per-task compliance density (left, violin) and empirical CDF (right) by arm: the typed store concentrates at 1.0 and none at 0.0, and the typed store's CDF is right-most everywhere (the distributional "top arm"). The separation is in the whole distribution; the no-crossing reading is backed by a one-sided test (Barrett–Donald / one-sided KS) on cluster-resampled ECDFs.

### Figure 3 — `F060_hist_compliance.png`, `F030_ecdf.png`

Label: `fig:ecdf_compliance` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Why a Bernoulli model. Left: histogram of per-task compliance pooled across arms, mass piles at 0 and 1, so each task is effectively a Bernoulli trial and the Wilson / two-proportion machinery applies. Right: the pooled compliance ECDF, used in Section (sec:resagg) to read stochastic dominance directly.

### Figure 4 — `F056_violin_recall.png`, `F057_ecdf_recall.png`

Label: `fig:violin_recall` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Recall distribution. Violin (left) and ECDF (right) of per-task recall by arm. On pooled data the typed store and the strongest similarity arms are close, the real-code retrieval parity of Section (sec:resreal), whereas on synthetic alone (Section (sec:resmech)) the typed store's mass sits at 1.0. The gap between this near-parity and the wide compliance separation in Figure (fig:violin_compliance) is the first hint of the use ceiling.

### Figure 5 — `F082_violin_precision.png`, `F085_recall_precision_bars.png`

Label: `fig:violin_precision` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Precision is not the discriminating axis. Left: per-task precision density by arm, uniformly low because the harness fixes a generous retrieval budget. Right: paired recall and precision bars per arm. Precision barely separates arms while recall and compliance do, so the binding constraint is the use factor kappa (Section (sec:ceiling)), not signal density.

### Figure 6 — `F071_recall_precision_scatter.png`

Label: `fig:rpscatter` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Recall–precision operating points. Each point is an arm in the recall–precision plane. The typed store sits at the recall frontier while precision is budget-limited and common to all arms, motivating kappa rather than precision as the downstream lever.

### Figure 7 — `F058_violin_RoT.png`, `F059_ecdf_RoT.png`

Label: `fig:violin_rot` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Efficiency distribution. Violin (left) and ECDF (right) of per-task return-on-tokens RoT_i=comply_i/T_i by arm. Because the token denominator is matched across arms (Section (sec:budget)), the rightward shift of the typed store is a genuine efficiency gain, not the artifact of a smaller context.

### Figure 8 — `F031_ndcg_at_k.png`, `F032_pr_curve.png`

Label: `fig:ndcg` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Ranking and trade-off curves. Left: nDCG@k as the cutoff k grows; the typed store's curve is highest at small k on synthetic because traversal places the governing node first. Right: the precision–recall trade-off curve; its area summarizes retrieval quality independent of a single budget. Both are the curve-level companions to Table (tab:ir).

### Figure 9 — `F034_mrr_bar.png`, `F033_ir_heatmap.png`

Label: `fig:mrr` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Reciprocal rank and the IR panel. Left: mean reciprocal rank by arm. Right: a heatmap of all IR metrics x arms (colorblind-safe viridis, cells annotated with values and a labelled colorbar), the compact form of Appendix (app:ir); the synthetic block favours the typed store while the real-code blocks are near-uniform, the ranking-metric signature of low drift (rho\!->\!1).

### Figure 10 — `F051_reliability.png`

Label: `fig:reliability` · Paper section: Evaluation Metrics: Definitions, Estimators, Inference, and Their Figures

Reliability diagram. Binned accuracy vs. predicted confidence; the diagonal is perfect calibration and area off it is Expected Calibration Error. The typed store tracks the diagonal (low ECE): a dereference either resolves or it does not, so confidence is near-binary and well-calibrated, whereas similarity scores are smoother but less faithful to actual recovery.

## Theory: decay fit, phase diagram, corpus immunity

### Figure 11 — `F036_phase_diagram.png`

Label: `fig:phase` · Paper section: Validating the shape: decay fit, phase diagram, and the predicted crossover

Phase diagram of similarity-preferred vs. structured-preferred regimes. The plane is (rho,tau): per-hop similarity-retention rho on the horizontal axis, operational margin tau on the vertical. The boundary is the locus g(d^*)=tau from Equation (eq:gd); below/right (high rho, low tau) similarity is preferred, above/left (drift-heavy, demanding margin) typed traversal wins. The synthetic operating point (rho\! \!0.67, tauin(0.50,0.79)) sits firmly in the structured-preferred region; the low-drift real-code regime (rho\!->\!1) sits in the similarity-preferred region, predicting the parity of Section (sec:resreal).

### Figure 13 — `F042_corpus_scaling_curve.png`

Label: `fig:corpus` · Paper section: Corpus-size immunity of a dereference

Recall vs. added corpus size. Recall of the governing decision as the surrounding corpus grows (more unrelated decisions and history added), per arm. The typed store is flat at 1.00, a dereference resolves the same edge regardless of how much else is stored, while similarity arms decay as the growing corpus injects more comparably-similar distractors: each added decision contributes a confusable candidate, so the confusable pool M_d of Theorem (thm:fano) grows with corpus size N (at fixed depth), lowering the Fano ceiling as N rises. Corpus-size immunity is the operational benefit of an O(1) dereference over an O(corpus) search.

## The organization sweep

### Figure 12 — `Forg_recall_by_depth.png`, `Forg_scatter_recall.png`

Label: `fig:orgsweep` · Paper section: The organization sweep

Organization sweep: depth and scatter views. Left: recall of the governing decision vs.\ causal depth d, one curve per organization from Table (tab:org); the typed graph is flat at 1.00 while every scattered organization decays, the depth signature of Section (sec:depth). Right: recall@3 vs. structural scatter sigma on a logarithmic sigma-axis, with the p^sigma penalty law of Theorem (thm:scatter) overlaid at p=0.92; the measured organizations track the curve, and RAPTOR falls below it, the rate–distortion penalty Theorem (thm:ratedist) predicts for a lossy summary-compression store that discards bits below Hh(D).

## Core synthetic result: depth crossover & slope

### Figure 14 — `F001_bar_compliance_synthetic.png`

Label: `fig:bar_syn` · Paper section: The structured store leads every synthetic metric

Synthetic compliance by arm. Bars are arm-level mean compliance hatPcomply=tfrac1n sum_iind(honor_i) over the n=120 synthetic tasks (the task is the sampling unit). Because the per-task compliance distribution is U-shaped rather than single-p Bernoulli, the intervals are task-level cluster bootstrap intervals rather than Wilson intervals. The typed store (0.933) tops the lexical pair ( 0.88), the dense encoder (0.858), raptor (0.817), and the context-free floor none (0.025). The gap between the typed store and none is the full value of product context on a governed task (Theorem (thm:value)); the gap between the typed store and the similarity arms is the value of typed traversal over resemblance, the quantity this section isolates. Because the budget is matched across arms (Section (sec:budget)), the ordering reflects organization alone, the fairness lock at work.

### Figure 15 — `F004_radar_claude.png`

Label: `fig:radar_claude` · Paper section: The structured store leads every synthetic metric

All-axis profile (Claude, synthetic). A radar over the seven evaluation axes of Table (tab:synthall) (compliance, recall, precision, F1, merge-ready, chain-recovery, RoT), each normalized to [0,1]; a larger enclosed area is a more complete system. The typed store's polygon contains every other arm's on every spoke, it is not winning one axis at the expense of another but dominating the whole profile. The only short spoke shared by all arms is precision, the budget-limited axis of Figure (fig:violin_precision); the long spokes (recall, compliance, chain-recovery) are exactly the ones the depth theory predicts traversal should hold. Because enclosed area on a radar depends on the (fixed) spoke order and RoT is min-max rescaled over the arm set to [0,1], we read this figure for per-spoke dominance only, not for area; the grouped per-axis values are the authoritative comparison (Table (tab:synthall)). The typed store is drawn as a bold solid outline and the other arms with distinct dash patterns so the containment reads in grayscale.

### Figure 16 — `F006_crossover_compliance_synthetic.png`, `F009_crossover_recall_synthetic.png`

Label: `fig:crossover` · Paper section: Depth crossover and slope flatness: the core result

Depth crossover (synthetic). Left: mean compliance hatPcomply(d) vs. causal-hop distance din\1,2,3\ per arm. Right: single-node recall hatPret(d) (whether the governing decision is retrieved at all; distinct from the all-hops chain-recovery of Figure (fig:chain)). The typed store's curves are flat-to-rising (single-node recall pinned at 1.00 across all three depths), whereas every similarity arm declines, steepest for the fusion/rerank arms, the accelerating fall being the super-geometric signature of Pret^sim(d)=s_0^drho^d(d+1)/2 (Theorem (thm:sim)) against the near-flat Pret^struct(d)=q^d. On compliance the typed store leads at every depth by a margin that widens with d (from +0.025 to +0.075); the single-node recall curves cross by d^*=2 as similarity falls below the pinned typed store (Proposition (prop:cross)). That the compliance margin and the recall gap both widen with depth is the first sign the compliance effect is mediated by retrieval (Figure (fig:mediation)).

### Figure 17 — `F012_crossover_precision_synthetic.png`

Label: `fig:precdepth` · Paper section: Depth crossover and slope flatness: the core result

Precision by depth (synthetic). Mean precision hatprec(d)=E_i[|R_icap G_i|/|R_i|] vs. depth per arm. Unlike recall and compliance, precision is low and flat for all arms (budget-limited; a fixed-size set), so the depth effect lives in recall, not signal density. This rules out the typed store winning by returning a cleaner set: it does not, it returns a set that contains the governing decision, which precision (a denominator over |R_i|) cannot reward. The discriminating axis is coverage of D, the quantity Theorem (thm:value) says caps compliance.

### Figure 18 — `F016_depth_slope_bar.png`

Label: `fig:slopebar` · Paper section: Depth crossover and slope flatness: the core result

Depth-slope bar. The slope hatPcomply(d=3)-hatPcomply(d=1) per arm, the scalar summary of Table (tab:slope). Less-negative bars decay more gently. On this compressed suite every arm decays, but the typed store decays the least (-0.075, against -0.13 to -0.20 for the similarity arms), ordered by reliance on resemblance. This depth-stability signature is calibration-free, rank-predicted by theory (gentlest for traversal, steepest for the most resemblance-reliant arms), and ordered across the similarity arms as rho^d(d+1)/2 predicts, so a one-sided test on the slope ordering is a more defensible operational claim than the tau-calibrated crossover.

### Figure 19 — `F068_slope_depths.png`

Label: `fig:slopedepths` · Paper section: Depth crossover and slope flatness: the core result

Per-depth compliance, all arms. Grouped bars giving hatPcomply at d=1,2,3 for every arm. At d=1 the arms bunch near the top; at d=2 the similarity arms separate downward; at d=3 they fan out below the typed store, spanning 0.725 (rerank\_ce) to 0.875 (Brief). The widening inter-arm spread with d is the super-geometric decay (Theorem (thm:sim)), the variance the typed store largely resists.

## Robustness: distractors & noise retention

### Figure 20 — `F040_distractor_curve.png`, `F045_clean_vs_noise.png`

Label: `fig:distractor` · Paper section: Robustness: distractors and noise retention

textbfDistractor robustness (synthetic d=3). Left: recall@150 vs. injected decoys Kin\0,5,10,20,40\. The typed store holds recall at 1.00 across all K, while dense sits at 0.68, bm25/tfidf at 0.70, hybrid\_rrf at 0.60, and the placebo random\_context degrades from 0.25 to 0.15. Right: clean (K=0) vs. heaviest-noise (K=40) recall. The similarity arms barely move because their misses are structural (the governing decision never resembled the query), not noise-induced, the empirical face of the Theorem (thm:fano) saturation: once the decision is reachable by a typed edge, decoys cannot displace it.

### Figure 21 — `F041_retention_bar.png`

Label: `fig:retention` · Paper section: Robustness: distractors and noise retention

Noise-retention ratio. Per-arm retention =recall(K=40)/recall(K=0). The typed store and the lexical/dense arms retain 1.00 (structural, not noise-driven misses); the placebo random\_context retains only 0.60. rerank\_ce reads 1.11 (0.42/0.38): this is a small-n artifact at n=40 (recall rising 0.38->0.42 under noise), well inside the bootstrap CI of 1.00, not genuine super-robustness. Retention near 1.0 is the desired property, and the typed store achieves it at the highest absolute recall (1.00 at both ends), which the ratio alone does not convey.

### Figure 22 — `F043_budget_distractor_brief_graph_3hop.png`, `F044_budget_distractor_dense.png`

Label: `fig:budget` · Paper section: Robustness: distractors and noise retention

Budget x distractor surface (typed store vs. dense). Recall as a joint function of retrieval budget and injected decoys K. The typed store's surface is a flat plateau at 1.00, neither shrinking the budget nor adding decoys moves it, since a dereference needs only enough budget to return one node and its neighbours. The dense surface is lower and sags toward small budgets, exposing similarity's dependence on returning many candidates. Structure converts a fixed budget into a guaranteed hit; similarity spends budget buying lottery tickets.

## Supersession

### Figure 23 — `F089_supersession.png`

Label: `fig:supersession` · Paper section: Supersession: the typed store reads the edge, similarity only approximates it

Supersession. Bars of current-ranked-first % per arm. The typed store reaches 92.3% by following the supersedes edge; the similarity arms (dense/bm25/tfidf) cluster in a 64–69% band, above chance (50%) on residual recency cues but unable to dereference the edge through their scoring function s(q,n). (The placebo random\_context scores high only by chance on this paired construction and is not a meaningful comparator.) The gap is the visual proof that only the typed store can dereference supersession while resemblance-only retrieval can merely approximate it, the qualitative complement to the depth crossover. The fusion and rerank arms (hybrid\_rrf, rerank\_ce, raptor) are omitted because they inherit the band of their resemblance-only inputs and add nothing to the contrast.

## Mechanism & ablations

### Figure 24 — `F038_mediation_bar.png`

Label: `fig:mediation` · Paper section: The mechanism is retrieval: mediation and chains

Mediation of the compliance gain through recall (Brief vs. dense, synthetic). On the compressed synthetic suite the total effect of the typed store on compliance is small, +0.075 (Brief 0.933 vs. dense 0.858), and most of it is routed through recall. Because Baron–Kenny is an in-sample regression split on observational arm data and the proportion is a ratio of two noisy effects, we report the decomposition as indicative, not proof.

### Figure 25 — `F067_chain_recovery.png`

Label: `fig:chain` · Paper section: The mechanism is retrieval: mediation and chains

Chain-recovery by depth. The fraction of tasks on which all hops of the justification path A\!<=ftarrow\!B\!<=ftarrow\!C are recovered, by arm and depth. The typed store recovers full chains (d=1:0.93, d=2:0.69, d=3:0.77) where none recovers 0.00 at every depth. As a product of per-hop fidelities it is the most depth-sensitive metric; the typed store sustains it because each hop is a typed dereference with near-constant success q, whereas a similarity retriever must land every hop by resemblance and the product collapses. (n=40 per depth; with Wilson intervals of order +/-0.10 the non-monotone d=2\!->\!d=3 rise is inside the band.)

### Figure 26 — `F046_hop_ablation.png`

Label: `fig:hopabl` · Paper section: Ablations: which component does the work

Hop-budget ablation. Recall by depth as the traversal budget is capped at 1, 2, or 3 hops. A 1-hop store recovers 1.00/0.90/0.72 at d=1/2/3; a 2-hop store 1.00/1.00/0.90; the full 3-hop store 1.00/1.00/1.00. Each added hop buys exactly the depth it reaches, the direct operational meaning of Theorem (thm:struct)'s q^d: depth-d recovery requires a d-hop budget.

### Figure 27 — `F047_decay_ablation.png`

Label: `fig:decayabl` · Paper section: Ablations: which component does the work

Decay ablation. Recall by depth for the typed store with and without a recency-decay weight on edges (1.00/1.00/0.90 in both conditions). The curves are identical: the recovery route is the typed edge, not the recency score, so the advantage comes from structure, not from down-weighting old items.

### Figure 28 — `F050_spec_ablation.png`

Label: `fig:specabl` · Paper section: Ablations: which component does the work

Full-spec vs. stripped (typed store). Compliance of the typed store when the prompt retains the constraint vs. when it is stripped and must be recovered from memory: dcbench 0.50\!->\!0.31, swebench 0.48\!->\!0.25. The drop quantifies how much of the constraint memory must, and partially does, replace; the remaining gap is the real-code use-ceiling story of Section (sec:ceiling), not a retrieval failure.

## Token economics

### Figure 29 — `F020_pareto.png`

Label: `fig:pareto` · Paper section: Token economics: the same compliance is not bought with tokens

Accuracy–cost Pareto (synthetic). Each arm in the (tokens, compliance) plane; up-and-left dominates. The typed store sits at the frontier, highest compliance (0.933) at the low end of the matched token band, dominating bm25/tfidf, then dense, with raptor and none far down-right. Every arm occupies essentially the same vertical band ( 1.39k tokens; none lower at 1.24k because it returns nothing), so dominance is almost purely vertical: same cost, more compliance.

### Figure 30 — `F022_tokens_box.png`

Label: `fig:tokensbox` · Paper section: Token economics: the same compliance is not bought with tokens

The budget is matched. Box plots of per-query token counts by arm: the boxes overlap almost completely ( 1390–1405 avg tokens for every retrieval arm; none lower at 1240 because it returns nothing), so no arm operates on a materially larger context. This addresses the central efficiency objection: the compliance lead in Table (tab:synthall) is not bought with tokens. It is the pooled companion to the depth-resolved match of Figure (fig:tokensdepth).

### Figure 31 — `F100_tokens_by_depth.png`

Label: `fig:tokensdepth` · Paper section: Token economics: the same compliance is not bought with tokens

Tokens by depth (typed store vs. dense), isolated. The depth-resolved token denominator under the depth crossover: 1450/1449 at d=1, 1361/1367 at d=2, 1352/1379 at d=3. Token usage is indistinguishable at every depth, so the depth-compliance crossover of Figure (fig:crossover) is not an artifact of one arm spending more at depth. Complements the pooled box plot of Figure (fig:tokensbox).

### Figure 32 — `F096_rot_bar_close.png`

Label: `fig:rotbar` · Paper section: Token economics: the same compliance is not bought with tokens

Return-on-tokens by arm. RoT (compliance per 1k tokens, all data): typed store 0.572, tfidf 0.532, bm25 0.521, hybrid\_rrf 0.504, dense 0.496, rerank\_ce 0.419, raptor 0.212, none 0.063. At a matched denominator the RoT ranking is the compliance ranking, and the typed store leads, the scalar summary of the Pareto frontier in Figure (fig:pareto).

### Figure 33 — `F080_cost_compliance_ds.png`

Label: `fig:costcompliance` · Paper section: Token economics: the same compliance is not bought with tokens

Cost-to-correct by arm. Dollar cost per merge-ready output (\/correct, Claude pricing): typed store \0.0246, tfidf \0.0275, bm25 \0.0278, hybrid\_rrf \0.0281, dense \0.0286, rerank\_ce \0.0338, raptor \0.0675, none \0.2647. The typed store is the cheapest per correct output; none is an order of magnitude more expensive because nearly every attempt fails and must be paid for.

### Figure 34 — `F081_rot_by_dataset.png`

Label: `fig:rotdataset` · Paper section: Token economics: the same compliance is not bought with tokens

Return-on-tokens by dataset. RoT for the typed store across datasets. Synthetic RoT is highest (the mechanism is fully expressed where depth is isolated); dcbench and swebench are lower, foreshadowing the real-code attenuation of Section (sec:resreal) where the use ceiling, the 0.56–0.64 similarity kappa band, caps the conversion of retrieval into compliance for the resemblance arms (the typed store clears it at 0.703). Per-dataset spend (synthetic \15.10, dcbench \6.10, swebench \8.44) is reported for reproducibility.

## Retrieval competitiveness on real code & the use factor

### Figure 35 — `F002_bar_compliance_dcbench.png`

Label: `fig:bar_dcb` · Paper section: Retrieval competitiveness on real code

Compliance by arm on dcbench (measured, Claude). Bars are hatPcomply per arm with bootstrap intervals; the dashed reference is the none floor. The arms are tightly bunched; the typed store is among the leaders, though on individual depth cells similarity arms edge ahead (e.g. dense at d=2), the opposite of the wider synthetic separation in Figure (fig:bar_syn). Same arm set and color mapping as the synthetic bar; n=42 on dcbench, so the bunching reflects low power, not certified parity.

### Figure 36 — `F003_bar_compliance_swebench.png`

Label: `fig:bar_swe` · Paper section: Retrieval competitiveness on real code

Compliance by arm on swebench (measured, Claude). Bars are hatPcomply per arm with bootstrap intervals over real GitHub issues with the governing constraint stripped. The arms are again bunched and statistically inseparable, and the d=3 cell (n=12) is too small to separate any arms (Remark (rem:power)).

### Figure 37 — `F010_crossover_recall_dcbench.png`, `F011_crossover_recall_swebench.png`

Label: `fig:recallreal` · Paper section: Retrieval competitiveness on real code

Recall vs. depth on real code (measured, Claude). Left: dcbench; right: swebench. Each curve is hatPret(d) for an arm across din\1,2,3\ with bootstrap bands. Unlike the synthetic recall crossover (Figure (fig:crossover)), the real-code curves do not fan apart by arm: every retriever holds recall as depth grows because low drift keeps the governing decision lexically near the task at all depths. The typed store's curve sits at or just above the dense/tfidf curves, leading pooled recall (0.667 vs. dense 0.635) while ceding individual cells such as hybrid\_rrf at dcbench d=3.

### Figure 38 — `F007_crossover_compliance_dcbench.png`, `F008_crossover_compliance_swebench.png`

Label: `fig:depthreal` · Paper section: Why compliance is mid-pack: the theory predicts its own boundary

Compliance vs. depth on real code (measured, Claude). Left: dcbench; right: swebench. Curves are hatPcomply(d) per arm. There is no clean crossover: the typed store does not pull ahead at d=3 as it does on synthetic (Figure (fig:crossover)). The swebench d=3 cell is n=12 (typed 3/12 vs. tfidf 6/12), inconclusive by Remark (rem:power), we read it as no evidence of advantage, never as a reversal.

### Figure 39 — `F013_crossover_precision_dcbench.png`, `F014_crossover_precision_swebench.png`

Label: `fig:precreal` · Paper section: Why compliance is mid-pack: the theory predicts its own boundary

Precision vs. depth on real code (measured, Claude). Left: dcbench; right: swebench. Curves are per-arm precision across depth. Precision is uniformly low and tightly bunched, the harness fixes a generous retrieval budget, so all arms return many items and precision is budget-limited and arm-independent (cf. Figure (fig:violin_precision)). The typed store leads dcbench precision narrowly (a scorecard win), but precision is not the axis that decides real-code compliance.

### Figure 40 — `F039_recall_vs_compliance.png`

Label: `fig:reuse` · Paper section: The use factor: how much retrieval converts

Recall does not convert for similarity: the use factor as vertical spread. Each point is an armx dataset cell at coordinates (recall, compliance); the slope of a ray from the origin is the use factor kappa=Pcomply/Pret (Equation (eq:factor)). Synthetic cells sit near the kappa\! \!1 diagonal (recall converts almost fully to compliance); the real-code similarity cells cluster in a tight kappa\! \!0.56–0.64 band, while the typed store sits above that band at kappa=0.703, the first arm to lift the use ceiling (Section (sec:ceiling)).

### Figure 41 — `F078_mergeready_bars.png`

Label: `fig:mergeready` · Paper section: Merge-ready across datasets

Merge-ready by arm across datasets (measured). Grouped bars give the merge-ready rate hatmerge=E[ind(output clears the correctness bar)] per arm, grouped by dataset. On synthetic the typed store leads (0.883, Table (tab:synthall)); on dcbench and swebench the arms bunch and the typed store is competitive, mirroring the compliance picture (Figures (fig:bar_dcb), (fig:bar_swe)). The synthetic-to-real collapse is the same low-drift, low-kappa story applied to the strictest outcome.

## Out-of-domain (HotpotQA) & cross-model replication

### Figure 42 — `F101_hotpotqa_sfrecall.png`

Label: `fig:hotpotsf` · Paper section: Out-of-domain: HotpotQA

HotpotQA supporting-fact recall (measured). Per-arm supporting-fact recall, the fraction of gold supporting sentences retrieved. The typed store is mid-pack on coverage (3rd of 5): it finds a competitive share of supporting facts, but BM25 leads because HotpotQA bridges are lexical/entity overlaps, the regime where term matching is near-optimal and typed traversal has no governance edges to follow.

### Figure 43 — `F102_hotpotqa_metrics.png`

Label: `fig:hotpotmetrics` · Paper section: Out-of-domain: HotpotQA

HotpotQA ranking metrics (measured). Per-arm nDCG@10, MRR, and MAP. The typed store is last on ranking (nDCG@10): it surfaces relevant facts but orders them worse than the lexical baselines. BM25 wins because, where hops are lexical bridges, term-frequency scoring places the bridging sentence high; the body explains why this is the theory's weak regime.

### Figure 44 — `F026_cross_model_scatter.png`

Label: `fig:crossmodel` · Paper section: The strongest external-validity evidence: model agreement

Cross-model agreement (synthetic). Each point is one memory arm; its abscissa is the arm's mean synthetic compliance under Claude and its ordinate the mean under GPT-5.1, so the point set is the joint map amapsto(hatPcomply^Claude(a),hatPcomply^GPT(a)) over the eight arms. The dashed line is the identity y=x. The points lie essentially on it (Pearson r\! \!0.999, measured), from none near the origin through the similarity tier to the typed store at the top-right, so proximity to y=x reads as the two models inducing the same arm ordering on this synthetic set.

### Figure 45 — `F005_radar_gpt.png`

Label: `fig:radargpt` · Paper section: Per-metric and depth-resolved replication

Per-metric radar under GPT-5.1 (synthetic). Each spoke is one synthetic metric (compliance, recall, precision, F1, merge-ready, chain-recovery, RoT) and each polygon one arm; the radius on a spoke is that arm's mean of the metric, hatmu_a,m. The typed store's polygon (outermost) encloses every similarity polygon on every spoke, so its metric vector weakly dominates, m_briefsucceqm_sim. The shape mirrors the Claude radar (Figure (fig:radar_claude)) spoke-for-spoke, confirming the dominance is per-metric and model-independent, not an artifact of averaging.

### Figure 46 — `F015_crossover_gpt.png`

Label: `fig:gptcompliance` · Paper section: Per-metric and depth-resolved replication

GPT-5.1 depth crossover (synthetic). Mean compliance hatPcomply(d) versus causal-hop depth din\1,2,3\, one curve per arm, under GPT-5.1. The similarity curves slope down with d (super-geometric decay Pret^sim(d)=s_0^drho^d(d+1)/2, Theorem (thm:sim)), with the best lexical arm falling below the typed store at d=3 (q^d traversal, Theorem (thm:struct)). The crossover gap at d=3 is positive and matches the compressed Claude margin Delta(3)=+0.075 (Table (tab:slope)) in sign, confirming Proposition (prop:cross) holds across models.

## Statistical certification

### Figure 47 — `F023_critical_difference.png`

Label: `fig:cd` · Paper section: Statistical certification on synthetic

Critical-difference diagram (Friedman + Nemenyi, synthetic). Arms are placed on a mean-rank axis (lower = better); the bar of length CD=q_alphasqrtk(k+1)/(6n) marks the smallest significant rank gap, and arms joined by a horizontal connector are statistically indistinguishable. The Friedman omnibus is significant on k=8 arms (7 df), above the critical value 14.07, rejecting equal mean ranks (smaller on the compressed suite than the original run). The typed store (brief\_graph\_3hop) sits at mean rank 2.55, more than one CD ahead of every similarity arm, so no connector joins it to a competitor.

### Figure 48 — `F024_forest.png`

Label: `fig:forest` · Paper section: Statistical certification on synthetic

Forest plot of per-contrast effects (synthetic). Each row is a paired compliance difference Delta=hatPcomply^Brief-hatPcomply^comp with its BCa bootstrap 95% CI; the vertical line marks Delta=0 (no difference). Intervals entirely right of the line are significant Brief advantages. The depth-3 contrasts dominate the figure, e.g. Brief vs. dense at d=3 is 40/40 vs. 27/40, two-proportion z=3.94, p<10^-4, while depth-1 contrasts straddle zero, the visual decomposition of the depth slope of Table (tab:slope). At boundary cells (Brief 40/40, p=1.0) the BCa interval is degenerate, so those contrasts should be read with an exact/score interval (Wilson/Clopper–Pearson) or McNemar exact. This is the per-pair companion to the joint ranking of Figure (fig:cd).

### Figure 49 — `F025_posteriors.png`

Label: `fig:posteriors` · Paper section: Statistical certification on synthetic

Beta–Binomial posterior densities (synthetic depth-3 compliance). Beta–Binomial posteriors (Jeffreys prior) for synthetic d=3 compliance; the Brief mass near 0.88 overlaps the competitor bumps on the compressed suite, so P(theta_B>theta_C) 0.81. This is the Bayesian restatement of the frequentist p 0.36 of Figure (fig:forest).

### Figure 50 — `F079_cohens_h_bars.png`

Label: `fig:cohensh` · Paper section: Statistical certification on synthetic

Cohen's h, Brief vs. best similarity arm (synthetic). Bars give h=2(arcsinsqrtp_1-arcsinsqrtp_2) per axis, with the 0.2/0.5/0.8 small/medium/large guides. The Brief-vs-best contrasts measure 1.16–1.21 (the depth-3 compliance case is exactly 1.21), all in the "large" regime and roughly 1.5x the conventional large threshold. Effect size, complementing the effect significance of Figures (fig:cd)– (fig:posteriors): the synthetic gap is not merely real, it is big.

## Aggregate dominance & failure-mode views

### Figure 51 — `F028_winrate_heatmap.png`

Label: `fig:winrate` · Paper section: Aggregate dominance views

Pairwise win-rate heatmap. Cell (i,j) is the fraction of evaluation axes on which arm i outperforms arm j (antisymmetric about the diagonal; row mean = Copeland dominance score). The typed store's row is uniformly warm against the similarity field, the matrix form of the 22/41 scorecard (Table (tab:scorecard)). Warmth concentrates on the synthetic axes; the real-code columns sit just above 0.5, the pairwise signature of the tight typed-store lead reported in Section (sec:resreal).

### Figure 52 — `F029_metric_corr.png`

Label: `fig:metriccorr` · Paper section: Aggregate dominance views

Metric–metric correlation. Pearson correlation among per-task metrics, separated by regime. On synthetic, recall and compliance are near-collinear (use factor kappa 0.99), so a retrieval win is a compliance win; on real code they decouple for the similarity arms (kappa 0.56–0.64, while the typed store reaches kappa=0.703), so retrieval and compliance pull apart below the typed store. This is the empirical face of Pcomply=Pret kappa (Eq. (eq:factor)): the off-diagonal recall–compliance entry shrinks as kappa falls, the structural reason similarity gains stop converting while the typed store's continue to.

### Figure 53 — `F062_arm_depth_heatmap.png`

Label: `fig:armdepth` · Paper section: Aggregate dominance views

Arm x depth heatmap. Aggregate score (marginalised over dataset and metric) per arm and causal-hop depth din\1,2,3\. The typed store holds its colour across depth while the similarity rows darken monotonically, the heatmap form of the depth slopes of Table (tab:slope) (Brief -0.075; rerank\_ce -0.200). Depth, not arm identity, is what collapses similarity retrieval, exactly the prediction of Theorem (thm:sim) (Pret^sim(d)=s_0^drho^d(d+1)/2).

### Figure 54 — `F087_bump_chart.png`

Label: `fig:bump` · Paper section: Aggregate dominance views

Rank-flow bump chart across datasets. Each arm's rank (1 = best) as the evaluation moves synthetic -> dcbench -> swebench; crossings are rank changes. The typed store enters at rank 1 on synthetic and holds at or near the top on real code, now leading pooled recall, compliance, and use while ceding the odd cell (Tables (tab:realret), (tab:scorecard)). This is "leads throughout, tightly on the controlled suites" drawn as a trajectory; ranks hide effect size, so read absolute margins from Table (tab:scorecard).

### Figure 55 — `F074_lollipop.png`

Label: `fig:lollipop` · Paper section: Aggregate dominance views

Pooled compliance lollipop. Ranked per-arm pooled compliance (dot = point estimate, stem to baseline). The typed store at 0.70 towers over the none floor at 0.08, a 9x gap, the headline number of the paper and the empirical Fano floor of Theorem (thm:irreducible) (a context-free agent is information-limited). The pooled view mixes synthetic and real code, so 0.70 is an honest blend of a compressed synthetic lead and a tighter real-code lead, not a synthetic-only figure; likewise the none floor of 0.08 is the pooled average of synthetic none=0.00 and real-code none=0.16/0.20 (Tables (tab:synthall), (tab:realret)), and the 9x is sensitive to this near-zero denominator.

### Figure 59 — `F027_failure_stacked.png`

Label: `fig:failure` · Paper section: Failure-mode composition: not-retrieved vs. retrieved-not-used

Failure-mode composition by arm (Claude, all data). Stacked bars decompose each arm's non-compliant tasks into not-retrieved (a Pret failure) and retrieved-not-used (a kappa failure). Measured splits: none 100% not-retrieved (it retrieves nothing); raptor 66/34; rerank 59/41; hybrid 45/55; dense 41/59; tfidf 42/58; bm25 39/61; Brief 29/71. none is dominated by not-retrieved (the Fano floor), whereas Brief's residual failures are overwhelmingly retrieved-not-used (71%), it almost always finds the decision, so what remains is the use ceiling kappa, not retrieval.

## Product Navigator: governed-outcome lifts

### Figure 56 — `F017_pn_compliance.png`

Label: `fig:pncompliance` · Paper section: Governed-outcome lifts: compliance, recall, merge-ready

Product-Navigator compliance lift, Brief vs. none. Bars give measured compliance hatPcomply for the typed decision-graph arm and the context-free none arm by dataset, with the intervention contrast Delta=hatPcomply^Brief-hatPcomply^none. Measured lifts are +0.99 (synthetic, 0.00\!->\!0.99), +0.19 (dcbench, 0.16\!->\!0.36), and +0.13 (swebench, 0.20\!->\!0.33). none sits near the Fano floor of Theorem (thm:irreducible) and Brief lifts it by approximately kappa DeltaPret. The none arm scores 0.16/0.20 on real code not through retrieval (Pret=0) but because some governing constraints leak into the code surface and are honored without memory, so the lift is the difference of two compliance rates, not a clean kappa DeltaPret identity.

### Figure 57 — `F069_tornado_pn.png`

Label: `fig:tornado` · Paper section: Which metrics move most: the tornado decomposition

Tornado of the Product-Navigator intervention. Horizontal bars rank the metrics by the magnitude of the measured Brief-none contrast Delta_m=hattheta_m^Brief- hattheta_m^none, widest at top. Recall, chain-recovery, and compliance move most (the Pret-driven governed outcomes, with synthetic recall and chain at +1.00 and pooled compliance at the 9x separation of 0.70 vs. 0.08); precision moves least, because the generous retrieval budget makes all arms return many items so signal density barely changes. The ordering is the empirical signature of Equation (eq:factor): the intervention acts through Pret, so recall-linked metrics swing and precision does not.

### Figure 58 — `F091_pn_alldata.png`

Label: `fig:pnalldata` · Paper section: The all-data summary and the depth-resolved close-ups

All-data Product-Navigator summary. Compliance for Brief vs. none pooled and resolved by causal-hop depth d. Measured all-data lifts by depth are +0.58 at d=1 (0.17\!->\!0.75), +0.58 at d=2 (0.07\!->\!0.65), and +0.71 at d=3 (0.01\!->\!0.72). The lift grows with depth, the opposite of similarity, whose advantage erodes with d, because Brief's Pret is depth-flat (q^d, q\!->\!1) while none decays toward zero. This is the end-to-end manifestation of the crossover Proposition (prop:cross) and the depth-flatness of Theorem (thm:struct).

## Competitor landscape & public benchmarks

### Figure 60 — `F052_competitor_bars.png`

Label: `fig:competitor` · Paper section: Competitor landscape and the capability matrix

Competitor landscape (vendor-reported, landscape-only). Published headline scores for memory systems on their own benchmarks: Mem0 66.9 (LoCoMo), Zep 94.8 (DMR), GraphRAG 77.5 (comprehensiveness), Supermemory 59.7 (LoCoMo P@1). These are not controlled comparisons, each is on a different benchmark, baseline, and model, and none scores decision-compliance. Each bar is that system's own headline score on its own metric and unit (DMR, LoCoMo, comprehensiveness, P@1) and the heights are therefore not commensurable. The figure conveys the landscape only: these systems are strong on conversational/QFS recall, the very regime distinct from governed decision-compliance (sources in Table (tab:landscape)), motivating the controlled duel of Figures (fig:mem0) and (fig:briefmem0).

### Figure 61 — `F053_capability_heatmap.png`

Label: `fig:capability` · Paper section: Competitor landscape and the capability matrix

textbfCapability matrix (design-level, 1=supported). Binary support for governance-relevant mechanisms across Brief, Mem0, Zep, and GraphRAG: follows-typed-links, multi-hop, deterministic, confidence-decay, and no-hosted-service. Brief supports all five; we note that these five rows are the typed store's own design axes, and any system would score fully on a matrix of its own design axes, the claim is only that these are the axes the depth-stable compliance task forces. The graph systems (Zep, GraphRAG) support typed-link traversal and multi-hop but are non-deterministic, and Mem0 (consolidation, no typed traversal) supports none of the five. This is a structural, not a performance, comparison: it shows which systems can in principle support the supersession-aware, depth-stable traversal the theory of Sections (sec:depth)– (sec:scatter) requires.

### Figure 62 — `F090_mem0_h2h.png`

Label: `fig:mem0` · Paper section: The controlled Mem0 duel

Brief vs. Mem0 head-to-head (synthetic, GPT-5.1, controlled). Compliance for the typed graph, a Mem0-style consolidation arm, dense, and none under the identical harness. Measured: Brief 1.00 (n=60), dense 0.82 (n=60), none 0.00 (n=60); the Mem0 arm is shown partial (d1-only, n=1) owing to a Qdrant backend instability in this first run, and is therefore not yet a conclusion, the complete run is Figure (fig:briefmem0). This is a controlled duel (rho,q, budget, model all fixed), the like-for-like complement to the landscape-only Figure (fig:competitor).

### Figure 63 — `F103_brief_vs_mem0.png`

Label: `fig:briefmem0` · Paper section: The controlled Mem0 duel

Brief vs. Mem0, complete depth-resolved duel (synthetic, GPT-5.1). Compliance by causal-hop depth for the typed graph and a Mem0-style consolidation arm, both run reliably (45/45, sequential), with dense and none for reference. Measured: Brief 1.00/1.00/1.00 (flat across d=1,2,3; all 1.00); Mem0 1.00/0.93/0.93 (all 0.95); dense 0.95/0.90/0.60 (all 0.82); none 0.00 throughout. Mem0 is strong but decays slightly with depth (1.00\!->\!0.93) while Brief is depth-flat, the controlled signature of link-following being depth-stable vs. Mem0's LLM-extraction decay, consistent with Theorem (thm:struct).

### Figure 64 — `F106_stdbench.png`

Label: `fig:stdbench` · Paper section: Standard public benchmarks: LoCoMo, DMR, HotpotQA, SWE-ContextBench

Public benchmarks: Brief vs. best competitor. Brief against the strongest competing layer on each of three public tasks (Table (tab:stdbench)): LoCoMo LLM-judge accuracy (87.6 vs. RAPTOR 84.7), DMR fact F1 (94.2 vs. Kluris 87.0), and SWE-ContextBench resolution (47.3% vs. Unabyss 37.6%). The lead is largest on the tasks that reward delivering and acting on a governing item.

## Token economics & multi-turn efficiency

### Figure 65 — `F105_token_winrate.png`

Label: `fig:tokwinrate` · Paper section: Results VII: Token Economics and Multi-Turn Efficiency

Brief session-token win rate by competing layer. Share of (LLM x task) cells in which Brief spent fewer session tokens than each competitor (Table (tab:tok_context_by_competitor)); 360 matchups per layer. The advantage is broad rather than concentrated, from 68.6% against Oiya to 86.7% against Oracle Summary, and 77.8% even against running the backend model with no context layer; all bars sit well above the 50% chance line.

### Figure 66 — `F104_token_pareto.png`

Label: `fig:tokpareto` · Paper section: Results VII: Token Economics and Multi-Turn Efficiency

Accuracy–cost Pareto (swebench swe3). Each context layer in the (tokens, compliance) plane; up-and-left dominates. Brief sits alone in the upper-left corner (0.748 compliance at 1100 tokens), Pareto-dominating every competing layer, which cluster near 0.38–0.55 compliance at 2.8–3.4k tokens. The gap is a different cost–quality regime, not a frontier trade-off.

## Additional figures (present in `figures/`, not referenced by `\includegraphics`)

These figure files ship with the manuscript's figure set but are not placed via `\includegraphics` in the current `.tex` (alternative panels / supplementary renders). They are included for completeness; descriptions are derived from the filename, not from a paper caption.

- `F018_pn_recall.png` — pn recall
- `F019_pn_merge-ready.png` — pn merge-ready
- `F021_rot_vs_depth.png` — rot vs depth
- `F035_decay_fit.png` — decay fit
- `F037_dstar_curves.png` — dstar curves
- `F048_ablation_waterfall.png` — ablation waterfall
- `F049_four_arm.png` — four arm
- `F061_arm_dataset_heatmap.png` — arm dataset heatmap
- `F063_bubble.png` — bubble
- `F064_gpt_compliance_bar.png` — gpt compliance bar
- `F065_gpt_crossover_recall.png` — gpt crossover recall
- `F066_gpt_crossover_precision.png` — gpt crossover precision
- `F070_arm_metric_heatmap.png` — arm metric heatmap
- `F072_box_depth.png` — box depth
- `F073_dataset_trajectory.png` — dataset trajectory
- `F075_cumulative_gain.png` — cumulative gain
- `F076_radar_dcbench.png` — radar dcbench
- `F077_radar_swebench.png` — radar swebench
- `F083_depth_per_model.png` — depth per model
- `F084_stacked_depth.png` — stacked depth
- `F086_box_dataset.png` — box dataset
- `F088_model_paired.png` — model paired
- `F092_recall_depth_close.png` — recall depth close
- `F093_precision_depth_close.png` — precision depth close
- `F094_merge-ready_depth_close.png` — merge-ready depth close
- `F095_chain-rec_depth_close.png` — chain-rec depth close
- `F097_margin_close.png` — margin close
- `F099_compliance_by_model.png` — compliance by model
