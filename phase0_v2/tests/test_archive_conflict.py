"""Tests for archive_conflict module."""

import json
from pathlib import Path

import pytest

from phase0_v2.calibration.archive_conflict import archive_conflict


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a results directory with two model files."""
    records_a = [
        {"conflict_id": "emoji", "condition": "A", "response": "hello"},
        {"conflict_id": "emoji", "condition": "C", "response": "world"},
        {"conflict_id": "language_en_es", "condition": "A", "response": "hola"},
        {"conflict_id": "capitalization", "condition": "C", "response": "TEST"},
    ]
    records_b = [
        {"conflict_id": "emoji", "condition": "A", "response": "foo"},
        {"conflict_id": "language_en_es", "condition": "C", "response": "bar"},
    ]
    _write_jsonl(tmp_path / "model_a_results.jsonl", records_a)
    _write_jsonl(tmp_path / "model_b_results.jsonl", records_b)
    return tmp_path


def test_archives_matching_records(results_dir: Path) -> None:
    summary = archive_conflict("emoji", results_dir)

    assert summary == {
        "model_a_results.jsonl": 2,
        "model_b_results.jsonl": 1,
    }

    # Original files should not contain emoji records
    remaining_a = _read_jsonl(results_dir / "model_a_results.jsonl")
    assert len(remaining_a) == 2
    assert all(r["conflict_id"] != "emoji" for r in remaining_a)

    remaining_b = _read_jsonl(results_dir / "model_b_results.jsonl")
    assert len(remaining_b) == 1
    assert remaining_b[0]["conflict_id"] == "language_en_es"

    # Archived files should exist
    archive_dir = results_dir / "archived"
    assert archive_dir.is_dir()

    archived_a = _read_jsonl(archive_dir / "emoji_from_model_a_results.jsonl")
    assert len(archived_a) == 2
    assert all(r["conflict_id"] == "emoji" for r in archived_a)

    archived_b = _read_jsonl(archive_dir / "emoji_from_model_b_results.jsonl")
    assert len(archived_b) == 1


def test_no_matching_records(results_dir: Path) -> None:
    summary = archive_conflict("nonexistent_conflict", results_dir)

    assert summary == {}
    assert not (results_dir / "archived").exists()

    # Original files unchanged
    assert len(_read_jsonl(results_dir / "model_a_results.jsonl")) == 4
    assert len(_read_jsonl(results_dir / "model_b_results.jsonl")) == 2


def test_idempotent(results_dir: Path) -> None:
    summary1 = archive_conflict("emoji", results_dir)
    assert summary1 != {}

    # Run again — should be a no-op
    summary2 = archive_conflict("emoji", results_dir)
    assert summary2 == {}

    # Archived file should still have the original records (no duplicates)
    archived_a = _read_jsonl(results_dir / "archived" / "emoji_from_model_a_results.jsonl")
    assert len(archived_a) == 2


def test_only_matching_conflict_in_file(tmp_path: Path) -> None:
    """When all records in a file match, the file should be empty after archiving."""
    records = [
        {"conflict_id": "emoji", "condition": "A"},
        {"conflict_id": "emoji", "condition": "C"},
    ]
    _write_jsonl(tmp_path / "all_emoji_results.jsonl", records)

    summary = archive_conflict("emoji", tmp_path)
    assert summary == {"all_emoji_results.jsonl": 2}

    remaining = _read_jsonl(tmp_path / "all_emoji_results.jsonl")
    assert remaining == []


def test_no_results_files(tmp_path: Path) -> None:
    """Empty directory — no crash."""
    summary = archive_conflict("emoji", tmp_path)
    assert summary == {}


def test_preserves_record_content(results_dir: Path) -> None:
    """Archived records should be identical to the originals."""
    original = _read_jsonl(results_dir / "model_a_results.jsonl")
    emoji_originals = [r for r in original if r["conflict_id"] == "emoji"]

    archive_conflict("emoji", results_dir)

    archived = _read_jsonl(results_dir / "archived" / "emoji_from_model_a_results.jsonl")
    assert archived == emoji_originals
