"""Shared process-pool driver for the embarrassingly-parallel per-city builds.

The all-city jobs (hex-grid generation, NetCDF cube assembly) are independent
per city with disjoint outputs, so they parallelize cleanly across cores. This
module is the *one* place that dispatches workers, reports progress, and
aggregates failures — both ``notebooks/generate_agglomeration_hexgrids.ipynb``
and ``notebooks/build_city_cubes.ipynb`` call :func:`parallel_map`, so the two
share the same machinery instead of each carrying a near-identical pool loop.

**Worker protocol.** A *worker* is any picklable callable taking one work item
and returning a ``(item_id, status)`` tuple, where ``status`` is a string:

  - starts with ``"skip"`` (e.g. ``"skip-exists"``)  -> counted as **skipped**
  - starts with ``"FAIL"`` (e.g. ``"FAIL: ..."``)    -> counted as **failed**
  - anything else (e.g. ``"built"``, ``"ok"``)       -> counted as **done**

A well-behaved worker catches its own exceptions and returns a ``"FAIL: ..."``
status rather than raising — one bad city then can't tear down the whole pool.

**Process vs thread (the ``executor`` arg).** Pick by what the work is bound on:

  - ``executor="process"`` (default) — for **CPU-bound** work (the hex-grid and
    cube builds: shapely intersects, exactextract, NetCDF encoding). Sidesteps
    the GIL; size ``max_workers`` to cores.
  - ``executor="thread"`` — for **I/O / latency-bound** work (the GEE downloads:
    each task mostly *waits* on Earth Engine compositing + the HTTP fetch, which
    release the GIL). Threads share one process — so authenticate Earth Engine
    **once in the parent** and let workers reuse it — and ``max_workers`` can far
    exceed core count (16-32), bounded by the remote service's rate limit.

**Process start method.** ``executor="process"`` assumes the POSIX *fork* default
(Linux): a worker defined in a notebook cell, and the module-level constants /
DataFrames it closes over, are inherited by the children (read-only,
copy-on-write) without pickling. On spawn platforms (macOS/Windows) workers must
be importable top-level functions and the driver guarded by ``__main__``.
``executor="thread"`` has no such constraint (closures are fine).

**Shared mutable state is the caller's problem.** Disjoint *output files* are
safe; a *shared file every worker appends to* (e.g. the grid zone manifest) is
not — resolve/seed it in the parent first and hand workers read-only data.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple


def _flush_print(msg: str) -> None:
    """``print`` that flushes — so progress is live even when stdout is a file
    (block-buffered), e.g. a headless background run. The default ``print_fn``."""
    print(msg, flush=True)


@dataclass
class ParallelResult:
    """Tally returned by :func:`parallel_map`."""

    done: int = 0
    skipped: int = 0
    failed: int = 0
    failures: List[Tuple[object, str]] = field(default_factory=list)
    seconds: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return (f"done {self.done}, skipped {self.skipped}, "
                f"failed {self.failed} in {self.seconds:.0f}s")


def parallel_map(
    worker: Callable[[object], Tuple[object, str]],
    items: Iterable[object],
    *,
    max_workers: int = 6,
    label: str = "items",
    progress_every: int = 100,
    print_fn: Callable[[str], None] = _flush_print,
    executor: str = "process",
) -> ParallelResult:
    """Run ``worker`` over ``items`` in a pool, with progress + tally.

    Args:
        worker: callable taking one item, returning ``(item_id, status)`` (see
            the module docstring for the status-string protocol).
        items: the work items (materialized to a list so progress can show N/total).
        max_workers: pool size. For ``executor="process"`` tune to where CPU/disk
            saturates (~4-8); for ``executor="thread"`` it can far exceed cores.
        label: plural noun for the progress lines (e.g. ``"cities"``).
        progress_every: print a running tally every N completed items (0 = never).
        print_fn: where progress goes (default: a flushing ``print`` so logs are
            live under redirection; pass a logger/sink to redirect or mute).
        executor: ``"process"`` (CPU-bound, default) or ``"thread"`` (I/O-bound) —
            see the module docstring for which to pick.

    Returns:
        A :class:`ParallelResult` with the done/skipped/failed counts, the list of
        ``(item_id, status)`` failures, and the wall-clock seconds.
    """
    if executor not in ("process", "thread"):
        raise ValueError(f"executor must be 'process' or 'thread', got {executor!r}")
    Pool = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor

    items = list(items)
    total = len(items)
    res = ParallelResult()
    t0 = time.time()
    print_fn(f"{total} {label}; {max_workers} {executor} workers ...")

    done_n = 0
    with Pool(max_workers=max_workers) as ex:
        for item_id, status in ex.map(worker, items):
            done_n += 1
            if status.startswith("skip"):
                res.skipped += 1
            elif status.startswith("FAIL"):
                res.failed += 1
                res.failures.append((item_id, status))
                print_fn(f"  FAIL {item_id} {status}")
            else:
                res.done += 1
            if progress_every and done_n % progress_every == 0:
                print_fn(f"  {done_n}/{total} | done {res.done}, "
                         f"skipped {res.skipped}, failed {res.failed} "
                         f"| {time.time() - t0:.0f}s")

    res.seconds = time.time() - t0
    print_fn(f"\n{label}: {res}")
    return res
