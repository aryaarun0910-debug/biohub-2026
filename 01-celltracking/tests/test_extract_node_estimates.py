from pathlib import Path

import pytest

from scripts.win_bet.extract_node_estimates import extract_estimates


def test_extracts_sorted_positive_integer_metadata(tmp_path: Path):
    (tmp_path / "b.geff").mkdir()
    (tmp_path / "a.geff").mkdir()
    values = {"a.geff": 11.0, "b.geff": 17.0}

    got = extract_estimates(tmp_path, reader=lambda path: values[Path(path).name])

    assert list(got) == ["a", "b"]
    assert got == {"a": 11, "b": 17}


@pytest.mark.parametrize("value", [0.0, -1.0, 2.5])
def test_rejects_invalid_metadata(tmp_path: Path, value: float):
    (tmp_path / "a.geff").mkdir()
    with pytest.raises(ValueError, match="invalid estimated_number_of_nodes"):
        extract_estimates(tmp_path, reader=lambda _path: value)


def test_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no GEFF"):
        extract_estimates(tmp_path, reader=lambda _path: 1.0)
