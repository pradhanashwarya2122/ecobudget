import json
from pathlib import Path

import pytest

from ml_retriever.types import Passage, Requirement

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def corpus_rows():
    with (DATA / "corpus.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def tasks():
    return json.loads((DATA / "tasks.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def splits():
    return json.loads((DATA / "splits.json").read_text(encoding="utf-8"))


class TestCorpus:
    def test_expected_seed_size(self, corpus_rows):
        # Seed corpus; Phase 1's eventual target is 200-500 (see plan.md).
        # 86 original + 6 (train-coverage fix) + 4 (Phase 1 scale-up film
        # summary passages: Inception, Lion King, Finding Nemo, Jurassic Park).
        assert len(corpus_rows) == 186  # +90 Phase C entities

    def test_every_row_is_a_valid_passage(self, corpus_rows):
        for row in corpus_rows:
            # Constructing a Passage exercises its __post_init__ validation.
            Passage(
                passage_id=row["passage_id"],
                text=row["text"],
                byte_size=row["byte_size"],
                source_url=row["source_url"],
                embedding=row["embedding"],
                metadata=row["metadata"],
            )

    def test_byte_size_matches_actual_text(self, corpus_rows):
        for row in corpus_rows:
            assert row["byte_size"] == len(row["text"].encode("utf-8"))

    def test_every_passage_has_a_source_url(self, corpus_rows):
        for row in corpus_rows:
            assert row["source_url"].startswith("https://")

    def test_passage_ids_are_unique(self, corpus_rows):
        ids = [row["passage_id"] for row in corpus_rows]
        assert len(ids) == len(set(ids))


class TestTasks:
    def test_task_count_in_target_range(self, tasks):
        # Original reviewed target was 150-200. Raised for the Phase 1 scale-up
        # (Phase-1-lever-plan.md): ~90 balanced seeds x synthetic variants to
        # push the decomposer past its 44% plateau toward the >90% Phase 2
        # target. Upper bound guards against runaway generation.
        assert 300 <= len(tasks) <= 1800  # Phase C scale-up

    def test_seed_task_count(self, tasks):
        # 41 original + 7 (train-coverage fix, T42-T48) + 42 (Phase 1
        # scale-up, T49-T90).
        seeds = [t for t in tasks if not t["synthetic"]]
        assert len(seeds) == 344  # Phase C scale-up (T91+)

    def test_task_ids_are_unique(self, tasks):
        ids = [t["id"] for t in tasks]
        assert len(ids) == len(set(ids))

    def test_synthetic_tasks_reference_a_real_seed(self, tasks):
        seed_ids = {t["id"] for t in tasks if not t["synthetic"]}
        for t in tasks:
            if t["synthetic"]:
                assert t["seed_task_id"] in seed_ids

    def test_synthetic_variant_preserves_seed_ground_truth(self, tasks):
        by_id = {t["id"]: t for t in tasks}
        for t in tasks:
            if not t["synthetic"]:
                continue
            seed = by_id[t["seed_task_id"]]
            assert t["decomposed_requirements"] == seed["decomposed_requirements"]
            assert t.get("expected_answer") == seed.get("expected_answer")
            assert t.get("ground_truth") == seed.get("ground_truth")

    def test_every_task_has_at_least_one_requirement(self, tasks):
        for t in tasks:
            assert len(t["decomposed_requirements"]) >= 1

    def test_answer_type_is_a_supported_closed_set(self, tasks):
        answer_types = {
            "single_fact",
            "yes_no",
            "list",
            "comparison",
            "multi_part",
            "procedure",
            "narrative",
        }
        for t in tasks:
            assert t["answer_type"] in answer_types

    def test_every_answer_type_has_a_seed_task(self, tasks):
        seed_answer_types = {t["answer_type"] for t in tasks if not t["synthetic"]}
        assert seed_answer_types == {
            "single_fact",
            "yes_no",
            "list",
            "comparison",
            "multi_part",
            "procedure",
            "narrative",
        }

    def test_topic_diversity_covers_all_five_categories(self, tasks):
        """Regression guard: the corpus/tasks must span all five categories
        from project_overview (comparisons/product selection, travel,
        specification lookup, multi-attribute research, decision making,
        sustainability), not just phone/laptop comparisons. See the
        conversation where this was flagged and fixed."""
        topics = {t["topic"] for t in tasks}
        # phones/laptops = product comparison; travel = travel;
        # buildings/geography = specification lookup; countries =
        # multi-attribute research; running_shoes = decision making;
        # vehicles = sustainability; electronics = product selection.
        required_topic_groups = {
            "product_comparison": {"phones", "laptops"},
            "travel": {"travel"},
            "specification_lookup": {"buildings", "geography"},
            "multi_attribute_research": {"countries"},
            "decision_making": {"running_shoes"},
            "sustainability": {"vehicles"},
            "product_selection": {"electronics"},
        }
        for group_name, expected_topics in required_topic_groups.items():
            assert expected_topics & topics, (
                f"No task covers the '{group_name}' category "
                f"(expected one of {expected_topics} in {topics})"
            )


class TestCoverage:
    def test_every_task_requirement_has_a_corpus_passage(self, corpus_rows, tasks):
        corpus_keys = {
            Requirement(entity=row["metadata"]["entity"], attribute=row["metadata"]["attribute"]).key()
            for row in corpus_rows
        }
        for task in tasks:
            for req in task["decomposed_requirements"]:
                key = Requirement(entity=req["entity"], attribute=req["attribute"]).key()
                assert key in corpus_keys, (
                    f"{task['id']}: no passage for "
                    f"({req['entity']!r}, {req['attribute']!r})"
                )


class TestSplits:
    def test_splits_partition_all_tasks_exactly_once(self, tasks, splits):
        all_ids = {t["id"] for t in tasks}
        split_ids = splits["train"] + splits["val"] + splits["test"]
        assert sorted(split_ids) == sorted(all_ids)
        assert len(split_ids) == len(set(split_ids))

    def test_no_seed_group_is_split_across_partitions(self, tasks, splits):
        """Regression guard for the leakage the plan's review flagged:
        every paraphrase of a seed task must land in the same split as
        its seed."""
        id_to_split = {}
        for split_name, ids in splits.items():
            for task_id in ids:
                id_to_split[task_id] = split_name

        by_id = {t["id"]: t for t in tasks}
        for task in tasks:
            seed_id = task["id"] if not task["synthetic"] else task["seed_task_id"]
            assert id_to_split[task["id"]] == id_to_split[seed_id]

    def test_test_split_is_nontrivial(self, splits):
        assert len(splits["test"]) >= 10

    def test_every_answer_type_and_attribute_has_train_coverage(self, tasks, splits):
        """Regression guard for the Phase 2 train-coverage bug: seed-level
        splitting had left whole answer types and attribute slugs with zero
        training data, making them unlearnable. Every answer type (except
        procedure, whose only seed is reserved for test) and every attribute
        slug must appear in the train split. Mirrors assert_train_coverage()
        in make_splits.py."""
        # procedure (both the answer type and the attribute slug) is exempt:
        # its only seed is reserved for the frozen test split.
        procedure_exempt = {"procedure"}
        train_ids = set(splits["train"])

        train_answer_types = set()
        train_attributes = set()
        all_answer_types = set()
        all_attributes = set()
        for t in tasks:
            all_answer_types.add(t["answer_type"])
            attrs = {r["attribute"] for r in t["decomposed_requirements"]}
            all_attributes |= attrs
            if t["id"] in train_ids:
                train_answer_types.add(t["answer_type"])
                train_attributes |= attrs

        missing_types = (all_answer_types - train_answer_types) - procedure_exempt
        missing_attrs = (all_attributes - train_attributes) - procedure_exempt
        assert not missing_types, f"answer types missing from train: {sorted(missing_types)}"
        assert not missing_attrs, f"attribute slugs missing from train: {sorted(missing_attrs)}"
