import pytest

from lib.config.loader import _compute_derived_fields


def test_compute_step_size_from_overlap():
    config = {"windowing": {"window": 1024, "overlap": 0.75}}
    _compute_derived_fields(config)
    assert config["windowing"]["step_size"] == 256


def test_rejects_invalid_overlap():
    config = {"windowing": {"window": 1024, "overlap": 1.0}}
    with pytest.raises(ValueError, match="overlap"):
        _compute_derived_fields(config)
