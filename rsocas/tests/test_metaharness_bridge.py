"""Tests for the Meta-Harness bridge."""

from __future__ import annotations

import json

import pytest

from rsocas.adapters.metaharness_bridge import CombinatorCandidate, MetaHarnessBridge
from rsocas.contracts.combinators import ValidationSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge(tmp_path):
    """Create a MetaHarnessBridge with temp directories."""
    return MetaHarnessBridge(
        candidates_dir=str(tmp_path / "candidates"),
        archive_dir=str(tmp_path / "archive"),
    )


def _valid_candidate() -> CombinatorCandidate:
    """Simple valid combinator candidate."""
    return CombinatorCandidate(
        name="merge_summaries",
        code=(
            "def merge_summaries(chunks: list[str]) -> str:\n"
            '    """Merge summaries by concatenation."""\n'
            '    return " ".join(chunks)\n'
        ),
        type_signature="(chunks: list[str]) -> str",
        hypothesis="Simple concatenation baseline.",
    )


def _validation_snapshot() -> ValidationSnapshot:
    """A synthetic validation snapshot."""
    return ValidationSnapshot(
        task_types=("qa", "summarization"),
        input_size_range=(100, 10000),
        n_samples=50,
        mean_score=0.75,
        score_std=0.08,
        timestamp=1000.0,
    )


# ---------------------------------------------------------------------------
# Tests — CombinatorCandidate serialization
# ---------------------------------------------------------------------------


class TestCombinatorCandidate:
    def test_to_dict_roundtrip(self) -> None:
        """Serialize and deserialize a candidate."""
        candidate = _valid_candidate()
        data = candidate.to_dict()
        restored = CombinatorCandidate.from_dict(data)

        assert restored.name == candidate.name
        assert restored.code == candidate.code
        assert restored.type_signature == candidate.type_signature
        assert restored.hypothesis == candidate.hypothesis

    def test_from_dict_defaults(self) -> None:
        """Missing optional fields get defaults."""
        data = {"name": "foo", "code": "def foo(): pass"}
        candidate = CombinatorCandidate.from_dict(data)

        assert candidate.type_signature == ""
        assert candidate.hypothesis == ""

    def test_frozen(self) -> None:
        """CombinatorCandidate is immutable."""
        candidate = _valid_candidate()
        with pytest.raises(AttributeError):
            candidate.name = "other"


# ---------------------------------------------------------------------------
# Tests — MetaHarnessBridge validation
# ---------------------------------------------------------------------------


class TestValidateCandidate:
    def test_validate_valid_candidate(self, bridge: MetaHarnessBridge) -> None:
        """Simple valid function passes validation."""
        candidate = _valid_candidate()

        valid, message = bridge.validate_candidate(candidate)

        assert valid is True
        assert message == "ok"

    def test_validate_syntax_error(self, bridge: MetaHarnessBridge) -> None:
        """Code with syntax error fails validation."""
        candidate = CombinatorCandidate(
            name="bad_syntax",
            code="def bad_syntax(:\n    pass\n",
        )

        valid, message = bridge.validate_candidate(candidate)

        assert valid is False
        assert "Syntax error" in message

    def test_validate_unbounded_loop(self, bridge: MetaHarnessBridge) -> None:
        """Code with 'while True' fails the bounded check."""
        candidate = CombinatorCandidate(
            name="infinite_loop",
            code=(
                "def infinite_loop():\n"
                "    while True:\n"
                "        pass\n"
            ),
        )

        valid, message = bridge.validate_candidate(candidate)

        assert valid is False
        assert "Unbounded" in message or "unbounded" in message.lower()

    def test_validate_recursive_call(self, bridge: MetaHarnessBridge) -> None:
        """Code with direct recursion fails the bounded check."""
        candidate = CombinatorCandidate(
            name="recurse",
            code=(
                "def recurse(n):\n"
                "    return recurse(n - 1)\n"
            ),
        )

        valid, message = bridge.validate_candidate(candidate)

        assert valid is False
        assert "recursive" in message.lower()

    def test_validate_no_callable(self, bridge: MetaHarnessBridge) -> None:
        """Code that defines no callable fails."""
        candidate = CombinatorCandidate(
            name="no_func",
            code="x = 42\n",
        )

        valid, message = bridge.validate_candidate(candidate)

        assert valid is False
        assert "callable" in message.lower()

    def test_validate_execution_error(self, bridge: MetaHarnessBridge) -> None:
        """Code that raises during exec fails."""
        candidate = CombinatorCandidate(
            name="exec_fail",
            code="raise ValueError('boom')\n",
        )

        valid, message = bridge.validate_candidate(candidate)

        assert valid is False
        assert "Execution error" in message


# ---------------------------------------------------------------------------
# Tests — File I/O
# ---------------------------------------------------------------------------


class TestWriteAndLoadCandidates:
    def test_write_and_load_candidates(self, bridge: MetaHarnessBridge) -> None:
        """Round-trip through filesystem preserves candidate data."""
        candidate = _valid_candidate()

        path = bridge.write_candidate(candidate)

        assert path.exists()
        assert path.suffix == ".py"

        loaded = bridge.load_candidates()

        assert len(loaded) == 1
        assert loaded[0].name == "merge_summaries"
        assert "merge_summaries" in loaded[0].code
        assert loaded[0].type_signature == "(chunks: list[str]) -> str"
        assert loaded[0].hypothesis == "Simple concatenation baseline."

    def test_load_empty_directory(self, bridge: MetaHarnessBridge) -> None:
        """Empty candidates dir returns empty list."""
        loaded = bridge.load_candidates()

        assert loaded == []

    def test_write_multiple(self, bridge: MetaHarnessBridge) -> None:
        """Multiple candidates are written and loaded."""
        c1 = _valid_candidate()
        c2 = CombinatorCandidate(
            name="filter_short",
            code="def filter_short(chunks: list[str]) -> list[str]:\n    return [c for c in chunks if len(c) > 10]\n",
            type_signature="(chunks: list[str]) -> list[str]",
        )

        bridge.write_candidate(c1)
        bridge.write_candidate(c2)

        loaded = bridge.load_candidates()

        assert len(loaded) == 2
        names = {c.name for c in loaded}
        assert names == {"merge_summaries", "filter_short"}


# ---------------------------------------------------------------------------
# Tests — Archive
# ---------------------------------------------------------------------------


class TestArchiveCandidate:
    def test_archive_candidate(self, bridge: MetaHarnessBridge, tmp_path) -> None:
        """Archive writes JSON with validation data."""
        candidate = _valid_candidate()
        validation = _validation_snapshot()

        bridge.archive_candidate(candidate, validation, accepted=True)

        archive_files = list((tmp_path / "archive").glob("*.json"))
        assert len(archive_files) == 1

        data = json.loads(archive_files[0].read_text())

        assert data["candidate"]["name"] == "merge_summaries"
        assert data["accepted"] is True
        assert data["validation"]["mean_score"] == 0.75
        assert data["validation"]["n_samples"] == 50
        assert "archived_at" in data

    def test_archive_rejected(self, bridge: MetaHarnessBridge, tmp_path) -> None:
        """Rejected candidates are archived with accepted=False."""
        candidate = CombinatorCandidate(name="bad", code="x = 1\n")
        validation = _validation_snapshot()

        bridge.archive_candidate(candidate, validation, accepted=False)

        archive_files = list((tmp_path / "archive").glob("*.json"))
        data = json.loads(archive_files[0].read_text())

        assert data["accepted"] is False


# ---------------------------------------------------------------------------
# Tests — Skill generation
# ---------------------------------------------------------------------------


class TestGenerateSkillMd:
    def test_generate_skill_md(self, bridge: MetaHarnessBridge) -> None:
        """Returns non-empty markdown string with required sections."""
        result = bridge.generate_skill_md("Discover reduction combinators for QA.")

        assert isinstance(result, str)
        assert len(result) > 100
        assert "# Combinator Discovery Skill" in result
        assert "Pure function" in result
        assert "Bounded execution" in result
        assert "Standard signature" in result
        assert "Type annotations" in result
        assert "Discover reduction combinators for QA." in result
        assert "Anti-patterns" in result
        assert "while True" in result
