from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import lapack
from threadpoolctl import threadpool_info

from athletevalue.runtime import (
    DETERMINISTIC_ENV,
    THREAD_VARIABLES,
    numerical_threads,
    with_numerical_threads,
)


def _blas_threads() -> set[int]:
    np.linalg.cholesky(np.eye(2))
    lapack.dpotrf(np.eye(2))
    return {pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"}


needs_controllable_blas = pytest.mark.skipif(
    not _blas_threads(), reason="no BLAS pool threadpoolctl can control (for example Accelerate)"
)


@needs_controllable_blas
def test_deterministic_mode_limits_blas_to_one_thread(monkeypatch):
    monkeypatch.setenv(DETERMINISTIC_ENV, "1")
    with numerical_threads():
        assert _blas_threads() == {1}


@needs_controllable_blas
def test_decorator_limits_threads_and_passes_arguments(monkeypatch):
    monkeypatch.setenv(DETERMINISTIC_ENV, "yes")

    @with_numerical_threads
    def add(a: int, *, b: int) -> tuple[int, set[int]]:
        return a + b, _blas_threads()

    assert add(1, b=2) == (3, {1})


def test_without_the_variable_thread_counts_are_untouched(monkeypatch):
    monkeypatch.delenv(DETERMINISTIC_ENV, raising=False)
    before = _blas_threads()
    with numerical_threads():
        assert _blas_threads() == before


def test_import_sets_thread_variables_before_numpy_loads():
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARIABLES}
    env[DETERMINISTIC_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    code = (
        "import os, athletevalue, numpy; "
        f"print(','.join(os.environ.get(v, '') for v in {THREAD_VARIABLES!r}))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert output == ",".join("1" for _ in THREAD_VARIABLES)
