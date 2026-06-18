# membench

Benchmark framework for the paper *Depth, Not Length: Why Coding Agents Fail and How
Structured Memory Fixes It*.

Tests whether structured memory (Brief) recovers deep causal context that
similarity-based retrieval cannot, across three datasets and five memory arms.

---

## Quick start

### 1. Requirements

Python **3.10 or higher** is required (the `mcp` package used by the Brief arm
needs 3.10+).

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Copy `.env.example` to `.env` and fill in your keys, then:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # required — main model
export BRIEF_OAUTH_TOKEN="..."               # required for the brief arm only
# export OPENAI_API_KEY="..."               # not yet implemented — see below
```

Or source the file directly:

```bash
source .env
```

### 3. Pre-seed Brief (brief arm only)

The `brief` arm's `write()` is a documented no-op — Brief's ingestion path is
interactive and approval-gated, so it cannot be called unattended inside a
benchmark loop. The workspace must be **pre-seeded with the task decisions before
running**. Follow Brief's own seeding guide (or adapt `benchmark/seed.ts` from the
dcbench repo) to load the decisions in `benchmarks/data/dcbench/decisions.json` into
your Brief workspace before firing a run that includes the `brief` arm.

If you skip this step, the brief arm will run but retrieve nothing, producing a
floor-level compliance score (which is a valid result — it means Brief wasn't given
anything to find, not that the arm is broken).

### 4. Run

```bash
python run.py
```

Results are written to `results/all_runs.jsonl` row-by-row as they complete
(crash-safe — if the run stops mid-way, you keep what finished).

### 5. Generate the four tables

```python
from scoring.tables import load_rows, headline_table, model_robustness_table, \
    depth_crossover_table, ablation_table
import json

rows = load_rows()
print(json.dumps(headline_table(rows), indent=2))
print(json.dumps(depth_crossover_table(rows), indent=2))
print(json.dumps(ablation_table(rows), indent=2))
print(json.dumps(model_robustness_table(rows), indent=2))
```

---

## Repo layout

```
membench/
  adapters/       none, fullcontext, mem0, brief, random_context
  agent/          runner.py (the task loop), llm_client.py
  benchmarks/     dcbench.py, swebench.py, longmemeval.py, depth_labels.jsonl
                  data/  (dcbench, swebench, longmemeval data files)
  configs/        fairness lock — same model/budget/tool across all arms
  scoring/        compliance.py, correctness.py, cost.py, tables.py
  results/        raw JSONL + summary tables (git-ignored)
  run.py          entry point
```

---

## The five arms

| Arm | What it does |
|-----|-------------|
| `none` | No memory — floor control |
| `fullcontext` | Whole corpus truncated to budget — "just buy tokens" control |
| `mem0` | Similarity top-k to budget (TF cosine, stdlib only) |
| `brief` | Graph walk via Brief MCP — needs `BRIEF_OAUTH_TOKEN` + pre-seeded workspace |
| `random_context` | Random nodes to Brief's exact budget — proves structure beats budget |

---

## The three datasets

| Dataset | Failure mode tested | Spec-stripped? |
|---------|--------------------|----|
| `dcbench` | Decisions invisible in code | Yes — d=1,2,3 |
| `swebench` | Cross-file constraint in remote module | Yes — d=1,2 |
| `longmemeval` | Temporal supersession (temporal + knowledge-update splits only) | No |

---

## Known gaps (flag before publishing numbers)

### depth_labels.jsonl — single-annotator only
`benchmarks/depth_labels.jsonl` was produced by an automated single-pass heuristic
(`dual_annotated: false` on every row). CLAUDE.md requires two independent annotators
tracing each path, with tasks disagreeing by more than one hop dropped or rewritten.
**Do not use the depth-crossover table for published numbers until a real dual-annotation
pass is done.** The headline table and ablation table are not affected by this gap.

### brief arm — Python ≥3.10 required
The `mcp` package used in `adapters/brief.py` requires Python 3.10+. The arm will
import-error on Python 3.9. Everything else runs on 3.9.

### Model-robustness table — GPT and open-weight not implemented
`configs/models.json` records GPT-5.1 and Llama 4 Maverick as the robustness-row
models, but their `agent/llm_client.py` backends have not been built yet
(`implemented: false`). `run.py` skips those cells with a printed notice rather than
crashing. The robustness table will only have the Claude row until those backends are
added.

---

## Running tests

```bash
pytest tests/
```

---

## Fairness invariant (what configs/ enforces)

Same model, same code-search tool (`none`), same retrieval `budget_tokens` per
dataset across every arm. Budget is passed *into* `retrieve()` by the caller — arms
never choose their own budget. This is the only variable being compared: memory
architecture.
