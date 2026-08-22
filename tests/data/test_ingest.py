from __future__ import annotations

import tempfile

import pytest

from src.data.etl import DataValidationError, run_etl
from src.data.generator import SMEDataGenerator


def test_etl_dry_run_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SMEDataGenerator(n_businesses=5, n_months=2, seed=10, output_dir=tmpdir)
        gen.generate()
        result = run_etl(tmpdir, dry_run=True)
        assert "businesses" in result
        assert "transactions" in result
        assert not result["businesses"].empty


def test_etl_fails_on_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            run_etl(tmpdir, dry_run=True)
