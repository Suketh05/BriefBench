# Retrieval Arms Catalog

A *memory arm* is one swappable retrieval/memory system under comparison. Every arm
implements the same `MemorySystem` contract — `write(items)` then
`retrieve(query, budget_tokens)` — and self-registers in a process-wide registry via the
`@register_arm` decorator. Memory architecture is the only variable that changes across
arms; the base model and the token budget are held fixed by the harness (the "fairness
lock"). See [`src/membench/retrieval/base.py`](../src/membench/retrieval/base.py) for the
contract and registry, and
[`src/membench/retrieval/__init__.py`](../src/membench/retrieval/__init__.py) for the
family overview.

The [README](../README.md) cites **22 memory arms**. This catalog enumerates all 22 as
they are actually registered in code, grouped by the `ArmFamily` each declares, and gives
for each arm: a short description, its retrieval **mechanism** (no-retrieval control /
lexical / dense / hierarchical / link-following), and its **role** (control, baseline,
the structured arm, or an external adapter).

Each arm name below is the exact string passed to `@register_arm(...)` — verified against
the source. The `follows_links` / `uses_embeddings` flags quoted are the decorator
arguments in code.

## How to read the role labels

- **Control** — a trivial floor/ceiling/negative reference, not a serious retriever.
- **Baseline** — a standard similarity retriever (lexical, dense, or hybrid) the
  structured arm is measured against.
- **Structured arm** — the typed decision-graph walk that models Brief's context graph
  (the system under test).
- **External adapter** — a thin adapter onto a third-party memory product, so it can run
  inside the same harness under the same budget.

## Arm count by family

| Family (`ArmFamily`) | Arms | Count |
|---|---|---|
| `CONTROL` | `none`, `full_context`, `random_context` | 3 |
| `LEXICAL` | `bm25`, `tfidf` | 2 |
| `DENSE` | `dense`, `raptor`, `hyde` | 3 |
| `HYBRID` | `hybrid_rrf`, `rerank_ce` | 2 |
| `GRAPH` | `brief_graph`, `brief_graph_1hop`, `brief_graph_3hop`, `brief_graph_decay`, `brief_live` | 5 |
| `EXTERNAL` | `mem0`, `supermemory`, `zep`, `graphrag`, `letta`, `cognee`, `langchain_vec` | 7 |
| **Total** | | **22** |

---

## Controls — `ArmFamily.CONTROL`

Defined in [`src/membench/retrieval/controls.py`](../src/membench/retrieval/controls.py).
None of the three perform relevance-based retrieval; they exist to bound the measurement.

### `none`
No memory at all: ingests nothing and retrieves nothing. This is the **floor control** —
it isolates the irreducible context-free error, the share of governed tasks an agent
cannot get right from the task surface alone. `follows_links=False`, `uses_embeddings=False`.
- **Mechanism:** no retrieval.
- **Role:** control (floor).

### `full_context`
Dumps the whole corpus in corpus order, truncated to the token budget — no relevance
ranking at all. It tests whether simply having everything (up to budget) is enough, and so
acts as a budget-limited ceiling reference for "no retrieval, just more context."
`follows_links=False`, `uses_embeddings=False`.
- **Mechanism:** no retrieval (ordered dump).
- **Role:** control.

### `random_context`
Fills the budget with randomly ordered items and deliberately ignores the query, so
selection cannot depend on relevance. This is the **budget-matched negative control**: it
spends the same retrieval budget as the structured arm but with random nodes, isolating
"budget, not structure." The shuffle is seeded for reproducibility.
`follows_links=False`, `uses_embeddings=False`.
- **Mechanism:** no retrieval (random fill).
- **Role:** control (negative). It is intentionally **not** in `DEFAULT_ARMS`; it is part
  of the dedicated `ABLATION_ARMS` set (`none`, `brief_graph_3hop`, `random_context`) run
  on the ablation datasets — see [`src/membench/harness.py`](../src/membench/harness.py).

---

## Lexical similarity baselines — `ArmFamily.LEXICAL`

Sparse, term-overlap retrievers. No embeddings (`uses_embeddings=False`),
`follows_links=False`.

### `bm25`
Ranks corpus items by Okapi BM25 similarity to the query, then packs the top results to
the token budget. The canonical strong lexical baseline.
Source: [`src/membench/retrieval/bm25.py`](../src/membench/retrieval/bm25.py).
- **Mechanism:** lexical.
- **Role:** baseline.

### `tfidf`
Ranks corpus items by TF-IDF cosine similarity to the query, then packs to budget. A
classic vector-space lexical baseline (distinct fit/scoring from BM25).
Source: [`src/membench/retrieval/tfidf.py`](../src/membench/retrieval/tfidf.py).
- **Mechanism:** lexical.
- **Role:** baseline.

---

## Dense and hierarchical baselines — `ArmFamily.DENSE`

Embedding-based retrievers (`uses_embeddings=True`), `follows_links=False`. By default they
use the deterministic offline hashing embedding provider so the offline pipeline is
reproducible.

### `dense`
Ranks corpus items by embedding cosine similarity to the query, then packs to budget. The
standard "plain RAG" dense-retrieval baseline; it can use a native cosine-top-k kernel with
a NumPy fallback.
Source: [`src/membench/retrieval/dense.py`](../src/membench/retrieval/dense.py).
- **Mechanism:** dense.
- **Role:** baseline (the head similarity baseline on real code).

### `raptor`
Clusters the corpus, builds centroid summary nodes, and retrieves over the collapsed tree
(RAPTOR-style). The number of cluster/summary nodes defaults to `round(sqrt(n))`. This is
the **hierarchical** retriever in the suite.
Source: [`src/membench/retrieval/raptor.py`](../src/membench/retrieval/raptor.py).
- **Mechanism:** hierarchical (cluster/summary tree over dense embeddings).
- **Role:** baseline.

### `hyde`
Embeds a *hypothetical* document generated from the query (HyDE), then ranks densely
against that synthetic embedding. The query-to-hypothetical-document expander defaults to a
deterministic template expander; the underlying ranking is the dense arm.
Source: [`src/membench/retrieval/hyde.py`](../src/membench/retrieval/hyde.py).
- **Mechanism:** dense (with query expansion).
- **Role:** baseline.

---

## Hybrid / reranking baselines — `ArmFamily.HYBRID`

Combine signals or add a second ranking stage (`uses_embeddings=True`,
`follows_links=False`).

### `hybrid_rrf`
Fuses the BM25 and dense rankings with Reciprocal Rank Fusion, then packs to budget. The
RRF constant `k` defaults to 60 (the original-paper value). The lexical+dense fusion
baseline.
Source: [`src/membench/retrieval/hybrid.py`](../src/membench/retrieval/hybrid.py).
- **Mechanism:** lexical + dense fusion.
- **Role:** baseline.

### `rerank_ce`
Retrieves first-stage candidates (the dense arm by default), reranks them jointly with a
pairwise scorer (a deterministic lexical-overlap reranker by default, standing in for a
cross-encoder), then packs to budget. The two-stage retrieve-then-rerank baseline.
Source: [`src/membench/retrieval/rerank.py`](../src/membench/retrieval/rerank.py).
- **Mechanism:** dense first-stage + reranking.
- **Role:** baseline.

---

## The structured arm — `ArmFamily.GRAPH`

The typed decision-graph walk that models Brief's context graph: it seeds by similarity and
then **follows stored typed links** (`follows_links=True`, `uses_embeddings=True` for seed
selection). The edge vocabulary is the governance relationship set
`constrains` / `implements` / `supersedes` defined in
[`src/membench/retrieval/graph/store.py`](../src/membench/retrieval/graph/store.py); the
traversal lives in
[`src/membench/retrieval/graph/traversal.py`](../src/membench/retrieval/graph/traversal.py)
and the usage-decay pass in
[`src/membench/retrieval/graph/decay.py`](../src/membench/retrieval/graph/decay.py). This
family is the system under test, and link-following is what distinguishes it from every
similarity baseline above.

### `brief_graph`
Seeds by embedding similarity, then follows typed links via bounded multi-hop traversal
(default `max_hops`, clamped to `[1, 3]` by the traversal layer; default `seeds=5`). This
is the base structured arm and the one wired into the curated "Claude + Brief" / "ChatGPT +
Brief" comparisons in
[`src/membench/agents/combos.py`](../src/membench/agents/combos.py).
Source: [`src/membench/retrieval/graph/arm.py`](../src/membench/retrieval/graph/arm.py).
- **Mechanism:** link-following (similarity seed + typed-edge traversal).
- **Role:** the structured arm.

### `brief_graph_1hop`
One-hop variant of `brief_graph`: follows a single typed link from each seed
(`max_hops=1`). Used to ablate traversal depth.
- **Mechanism:** link-following (1 hop).
- **Role:** the structured arm (hop ablation variant).

### `brief_graph_3hop`
Three-hop variant: follows up to three typed links from each seed (`max_hops=3`). This is
the headline structured configuration — it is the `brief_graph_3hop` entry in
`DEFAULT_ARMS` and the arm reported as "Brief" in the paper's main tables.
- **Mechanism:** link-following (up to 3 hops).
- **Role:** the structured arm (headline configuration).

### `brief_graph_decay`
Usage-aware variant: reinforces traversed links via the decay pass (`use_decay=True`), so
links the agent actually relies on are strengthened over time.
- **Mechanism:** link-following (with usage decay/reinforcement).
- **Role:** the structured arm (decay ablation variant).

### `brief_live`
Retrieves from a live, pre-seeded Brief workspace via the MCP `brief_search` tool rather
than the offline in-memory graph. It is the only non-deterministic, network-requiring graph
arm (`requires_network=True`, `deterministic=False`); `write` is a no-op because the
workspace is seeded out of band, and the OAuth token is read from an argument or
`BRIEF_OAUTH_TOKEN` (never hardcoded).
Source: [`src/membench/retrieval/graph/live.py`](../src/membench/retrieval/graph/live.py).
- **Mechanism:** link-following (live Brief backend).
- **Role:** the structured arm (live/real-backend variant).

---

## External adapters — `ArmFamily.EXTERNAL`

Thin adapters onto third-party memory products so each can run inside the same harness under
the same budget. Unless noted they are `requires_network=True`, `deterministic=False`, and
most build on the shared `KeyedExternalAdapter` base
([`src/membench/retrieval/external/_base.py`](../src/membench/retrieval/external/_base.py)).
The `follows_links` flag below reflects whether the underlying product is graph-based.

### `mem0`
Mem0-backed arm: semantic extraction plus search over the extracted memories.
`follows_links=False`, `uses_embeddings=True`.
Source: [`src/membench/retrieval/external/mem0_adapter.py`](../src/membench/retrieval/external/mem0_adapter.py).
- **Mechanism:** dense (semantic memory).
- **Role:** external adapter.

### `supermemory`
Supermemory-backed arm over an `add`/`search` client interface. `follows_links=False`,
`uses_embeddings=True`.
Source: [`src/membench/retrieval/external/supermemory_adapter.py`](../src/membench/retrieval/external/supermemory_adapter.py).
- **Mechanism:** dense.
- **Role:** external adapter.

### `zep`
Zep / Graphiti-backed **temporal knowledge-graph** memory arm. Because Graphiti is
graph-based it follows edges (`follows_links=True`, per the code comment),
`uses_embeddings=True`.
Source: [`src/membench/retrieval/external/zep_adapter.py`](../src/membench/retrieval/external/zep_adapter.py).
- **Mechanism:** link-following (temporal knowledge graph).
- **Role:** external adapter.

### `graphrag`
Microsoft GraphRAG-backed arm. Graph-based, so `follows_links=True`,
`uses_embeddings=True`.
Source: [`src/membench/retrieval/external/graphrag_adapter.py`](../src/membench/retrieval/external/graphrag_adapter.py).
- **Mechanism:** link-following (graph RAG).
- **Role:** external adapter.

### `letta`
Letta / MemGPT archival-memory-backed arm over an `add`/`search` client.
`follows_links=False`, `uses_embeddings=True`.
Source: [`src/membench/retrieval/external/letta_adapter.py`](../src/membench/retrieval/external/letta_adapter.py).
- **Mechanism:** dense (archival memory).
- **Role:** external adapter.

### `cognee`
Cognee-backed **knowledge-graph** memory arm. Graph-based, so `follows_links=True`,
`uses_embeddings=True`.
Source: [`src/membench/retrieval/external/cognee_adapter.py`](../src/membench/retrieval/external/cognee_adapter.py).
- **Mechanism:** link-following (knowledge graph).
- **Role:** external adapter.

### `langchain_vec`
LangChain FAISS vector-retriever arm. This adapter is purely local
(`requires_network=False`, `deterministic=True`), `follows_links=False`,
`uses_embeddings=True`.
Source: [`src/membench/retrieval/external/langchain_adapter.py`](../src/membench/retrieval/external/langchain_adapter.py).
- **Mechanism:** dense (FAISS vector store).
- **Role:** external adapter.

---

## Which arms the sweep runs by default

`DEFAULT_ARMS` in [`src/membench/harness.py`](../src/membench/harness.py) is the
representative, offline-runnable headline sweep:

```
none, bm25, tfidf, dense, hybrid_rrf, rerank_ce, raptor, brief_graph_3hop
```

`ABLATION_ARMS` (`none`, `brief_graph_3hop`, `random_context`) is the dedicated negative-
control ablation set, run on the ablation datasets only. The remaining arms — the other
graph variants (`brief_graph`, `brief_graph_1hop`, `brief_graph_decay`, `brief_live`),
`full_context`, and the seven external adapters — are registered and selectable by name but
are not part of the default offline sweep (the network/credentialed adapters need their own
environments). The eight `DEFAULT_ARMS` correspond to the "nine arms" framing in the paper
once `random_context` is added back from the ablation set.

---

## Provenance of any numbers cited here

Per the repository's integrity rules, repo-measured and paper-reported figures are kept
strictly separate and never presented as identical (poolings and conditions differ).

**Paper-reported** (from the paper digest at the operating point reported in the LaTeX;
cross-checked against `depth_not_length_FIXED.tex`, tables `tab:synthall` and `tab:realret`):
on real code (pooled dcbench+swebench, Claude) the structured arm `brief_graph_3hop`
("Brief") is reported first on recall (0.667), compliance (0.469), and use factor
(κ = 0.703), the only arm whose κ clears the 0.56–0.64 similarity band; on the synthetic
suite its reported compliance is 0.933 versus the `none` control's 0.025. These are
paper-reported figures.

**Repo-measured** (harness output committed in the repo, a different pooling than the paper
tables, so the values differ and must not be conflated): see
[`results/METRICS.md`](../results/METRICS.md) and
[`results/data/metrics_matrix.csv`](../results/data/metrics_matrix.csv), where, for example,
`recall@all` for Brief is recorded as 0.8441 (tier `measured`, `claude/all/d=all`) versus
0.7677 for the dense baseline. Treat those as the repo-measured cells and the paper-reported
numbers above as separate.
