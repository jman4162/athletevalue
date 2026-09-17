"""Process-level settings that affect numerical results."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from threadpoolctl import threadpool_limits

DETERMINISTIC_ENV = "ATHLETEVALUE_DETERMINISTIC"
"""Set to 1 to run BLAS and LAPACK on one thread.

Multithreaded BLAS may split a Cholesky factorization differently depending on the
thread count, which changes the last bits of ratings and cross-validation errors.
One thread makes repeated runs on the same machine and library build agree bit for
bit. It does not make different BLAS builds agree.

Two mechanisms apply, because no single one covers every build. Importing
``athletevalue`` before numpy sets the thread-count environment variables that
OpenBLAS, MKL, OpenMP and Apple Accelerate read when they load. After numpy is
loaded, ``threadpoolctl`` limits the pools it can reach (OpenBLAS, MKL, OpenMP);
it cannot reach Accelerate, so on macOS set the variable before starting Python.
"""

THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def deterministic_requested() -> bool:
    return os.environ.get(DETERMINISTIC_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def apply_thread_environment() -> None:
    """Set single-thread variables for BLAS libraries not yet loaded. Existing values win."""
    if not deterministic_requested() or "numpy" in sys.modules:
        return
    for variable in THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


@contextmanager
def numerical_threads() -> Iterator[None]:
    """Limit BLAS to one thread when ``ATHLETEVALUE_DETERMINISTIC`` is set; otherwise no-op."""
    if not deterministic_requested():
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def with_numerical_threads[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with numerical_threads():
            return function(*args, **kwargs)

    return wrapper
