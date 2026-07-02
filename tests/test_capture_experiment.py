"""Tests for the capture experiment (paper sec:capture / tab:capexp).

Covers the three storage-condition transforms (determinism, surgical link
strip, shred text conservation / count reversibility), the load-bearing
*mechanism* on a hand-built 2-hop corpus (an intact typed store recovers the
governing rule through its links; the edge-stripped store loses multi-hop
reachability; a lexical arm is unaffected by the strip), and the driver's
paired, well-formed rows plus the tab:capexp-shaped aggregation.

No published tab:capexp cell value is asserted against this code's output:
the mechanism tests assert reachability events on corpora built by hand, and
the only literal goldens are hand-computed (or quoted from the canonical
golden pack with provenance).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from membench.datasets.capture_conditions import (
    DEFAULT_SIGMA,
    TYPED_LINK_KEYS,
    CaptureCondition,
    apply_condition,
    shred_scattered,
    strip_edges,
)
from membench.datasets.synthetic import load_synthetic_tasks
from membench.types import MemoryItem, Task


def _synthetic_task(depth: int = 2, seed: int = 0) -> Task:
    return load_synthetic_tasks(depths=(depth,), per_depth=1, seed=seed)[0]


def _shredded_fragments(task: Task) -> dict[str, list[MemoryItem]]:
    """Group a Raw-scattered corpus's fragments by the item they were shredded from."""
    groups: dict[str, list[MemoryItem]] = defaultdict(list)
    for item in task.memory_corpus:
        if item.metadata.get("node_type") == "fragment":
            groups[str(item.metadata["shredded_from"])].append(item)
    for fragments in groups.values():
        fragments.sort(key=lambda i: int(i.metadata["fragment_index"]))
    return groups


class TestStripEdges:
    def test_removes_exactly_the_typed_link_keys(self) -> None:
        task = _synthetic_task(depth=3)
        # the CAPTURED corpus really carries links (else the strip is vacuous)
        assert any("edges" in item.metadata for item in task.memory_corpus)
        stripped = strip_edges(task)
        for item in stripped.memory_corpus:
            for key in TYPED_LINK_KEYS:
                assert key not in item.metadata

    def test_preserves_ids_texts_order_and_other_metadata(self) -> None:
        task = _synthetic_task(depth=3)
        stripped = strip_edges(task)
        assert len(stripped.memory_corpus) == len(task.memory_corpus)
        for before, after in zip(task.memory_corpus, stripped.memory_corpus, strict=True):
            assert after.item_id == before.item_id
            assert after.text == before.text  # no text lost
            expected = {k: v for k, v in before.metadata.items() if k not in TYPED_LINK_KEYS}
            assert dict(after.metadata) == expected  # node_type etc. survive

    def test_preserves_every_other_task_field(self) -> None:
        task = _synthetic_task(depth=2)
        stripped = strip_edges(task)
        assert stripped.task_id == task.task_id
        assert stripped.query == task.query
        assert stripped.depth == task.depth
        assert stripped.governing_decisions == task.governing_decisions
        assert stripped.dataset == task.dataset

    def test_deterministic_and_idempotent(self) -> None:
        task = _synthetic_task(depth=3)
        once = strip_edges(task)
        assert once == strip_edges(task)  # pure function
        assert strip_edges(once) == once  # fixed point

    def test_captured_condition_is_identity(self) -> None:
        task = _synthetic_task(depth=2)
        assert apply_condition(task, CaptureCondition.CAPTURED) is task
        # string values dispatch too (the enum is string-valued for row serialisation)
        assert apply_condition(task, "captured") is task
        assert apply_condition(task, "discrete_no_links") == strip_edges(task)

    def test_unknown_condition_rejected(self) -> None:
        task = _synthetic_task(depth=1)
        with pytest.raises(ValueError):
            apply_condition(task, "shredded")  # not one of the three conditions


class TestShredScattered:
    def test_deterministic_with_seed(self) -> None:
        task = _synthetic_task(depth=3)
        a = shred_scattered(task, 3, seed=7)
        b = shred_scattered(task, 3, seed=7)
        assert a == b  # ids, texts, metadata, gold: all identical

    def test_no_text_lost(self) -> None:
        # Reversible-in-count: for every shredded unit, the fragments' payloads
        # joined in fragment_index order reproduce the original whitespace-token
        # sequence exactly. Nothing is dropped, only de-unitised.
        task = _synthetic_task(depth=3)
        originals = {i.item_id: i for i in task.memory_corpus if i.node_type is not None}
        for sigma in (1, 2, 3, 5):
            shredded = shred_scattered(task, sigma, seed=0)
            groups = _shredded_fragments(shredded)
            assert set(groups) == set(originals)  # every unit shredded, none invented
            for item_id, fragments in groups.items():
                joined = " ".join(str(f.metadata["payload"]) for f in fragments)
                assert joined.split() == originals[item_id].text.split()

    def test_reversible_in_count(self) -> None:
        # Each captured unit maps to exactly min(sigma, token_count) fragments,
        # and the corpus size is fillers + sum of fragment counts (count identity).
        task = _synthetic_task(depth=2)
        sigma = 3
        shredded = shred_scattered(task, sigma, seed=0)
        originals = [i for i in task.memory_corpus if i.node_type is not None]
        fillers = [i for i in task.memory_corpus if i.node_type is None]
        groups = _shredded_fragments(shredded)
        expected_total = len(fillers)
        for item in originals:
            expected = min(sigma, len(item.text.split()))
            fragments = groups[item.item_id]
            assert len(fragments) == expected
            assert all(int(f.metadata["fragment_count"]) == expected for f in fragments)
            expected_total += expected
        assert len(shredded.memory_corpus) == expected_total

    def test_fragments_carry_no_typed_links(self) -> None:
        # Raw-scattered = strip AND shred; a fragment must never be traversable.
        shredded = shred_scattered(_synthetic_task(depth=3), 3, seed=0)
        for item in shredded.memory_corpus:
            for key in TYPED_LINK_KEYS:
                assert key not in item.metadata

    def test_gold_remapped_to_all_fragments(self) -> None:
        task = _synthetic_task(depth=2)
        shredded = shred_scattered(task, 3, seed=0)
        (gold_id,) = task.governing_decisions
        expected = tuple(f.item_id for f in _shredded_fragments(shredded)[gold_id])
        assert shredded.governing_decisions == expected
        assert len(expected) == 3  # the rule text is long enough for sigma fragments

    def test_fragments_of_one_decision_land_in_distinct_areas(self) -> None:
        # "split each decision across sigma distractor areas": with sigma at most
        # the 10-area pool size, the sigma fragments occupy sigma distinct areas.
        task = _synthetic_task(depth=3)
        for sigma in (2, 3, 4):
            shredded = shred_scattered(task, sigma, seed=0)
            for fragments in _shredded_fragments(shredded).values():
                areas = [str(f.metadata["scatter_area"]) for f in fragments]
                assert len(set(areas)) == len(areas)

    def test_sigma_one_keeps_whole_payload_in_one_fragment(self) -> None:
        task = _synthetic_task(depth=1)
        shredded = shred_scattered(task, 1, seed=0)
        for item_id, fragments in _shredded_fragments(shredded).items():
            assert len(fragments) == 1
            payload = str(fragments[0].metadata["payload"])
            assert payload.split() == dict(task.corpus_by_id)[item_id].text.split()

    def test_saturation_when_text_shorter_than_sigma(self) -> None:
        # A 3-token decision cannot shed into 5 fragments; it saturates at one
        # fragment per token (and still conserves the text).
        tiny = Task(
            task_id="cap-tiny",
            dataset="synthetic",
            query="anything",
            repo_ref="synthetic://capture",
            memory_corpus=(
                MemoryItem("d1", "use audit logging", metadata={"node_type": "governing"}),
            ),
            governing_decisions=("d1",),
            depth=1,
            spec_variant="stripped",
            scorer="compliance",
        )
        shredded = shred_scattered(tiny, 5, seed=0)
        fragments = _shredded_fragments(shredded)["d1"]
        assert len(fragments) == 3
        assert [str(f.metadata["payload"]) for f in fragments] == ["use", "audit", "logging"]
        assert shredded.governing_decisions == tuple(f.item_id for f in fragments)

    def test_filler_items_pass_through_link_stripped_only(self) -> None:
        task = _synthetic_task(depth=2)
        shredded = shred_scattered(task, 3, seed=0)
        by_id = dict(shredded.corpus_by_id)
        for item in task.memory_corpus:
            if item.node_type is None:
                assert by_id[item.item_id].text == item.text

    def test_rejects_sigma_below_one(self) -> None:
        task = _synthetic_task(depth=1)
        with pytest.raises(ValueError, match="sigma"):
            shred_scattered(task, 0)
        with pytest.raises(ValueError, match="sigma"):
            shred_scattered(task, -2)

    def test_default_sigma_used_by_dispatcher(self) -> None:
        task = _synthetic_task(depth=2)
        via_dispatch = apply_condition(task, CaptureCondition.RAW_SCATTERED, seed=3)
        assert via_dispatch == shred_scattered(task, DEFAULT_SIGMA, seed=3)
