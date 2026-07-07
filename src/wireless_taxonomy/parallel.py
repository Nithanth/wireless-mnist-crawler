"""Bounded, order-preserving parallel map for I/O-bound pipeline stages.

The pipeline's LLM and PDF-fetch stages are embarrassingly parallel: each
paper is classified/extracted independently (temperature 0, content-addressed
caching), so concurrency changes wall-clock time but not results. The
constraints that shape this helper:

* **Order preservation** — results are yielded in input order so downstream
  DB writes and progress logs are deterministic and identical to a
  sequential run.
* **Bounded in-flight window** — at most ``window`` items are submitted at a
  time, keeping memory bounded even when items carry PDF bytes.
* **Single-writer discipline** — workers only do network I/O (LLM calls,
  PDF fetches); the consumer (main thread) performs all SQLite writes.
  Item preparation also happens lazily in the caller's thread, so an items
  generator may safely read from SQLite.
* **Sequential fallback** — ``workers <= 1`` runs inline with no thread pool,
  preserving exact legacy behaviour.

Exceptions are yielded (not raised) so the consumer decides how each failure
is handled — e.g. ``CreditExhaustedError`` checkpoints the cache and aborts,
while a transient network error falls back per-paper.
"""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    window: int | None = None,
    task_timeout: int | None = None,
) -> Iterator[tuple[T, R | None, Exception | None]]:
    """Apply ``fn`` to ``items`` with ``workers`` threads, yielding in order.

    Yields ``(item, result, error)`` triples in input order: exactly one of
    ``result``/``error`` is non-None. With ``workers <= 1`` this degenerates
    to a plain sequential loop (no threads).

    ``task_timeout`` is the maximum seconds to wait for a single task before
    cancelling it and yielding a TimeoutError. If None, waits indefinitely.
    """
    if workers <= 1:
        for item in items:
            try:
                yield item, fn(item), None
            except Exception as exc:
                yield item, None, exc
        return

    window = window or workers * 2
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight: deque = deque()
        for item in islice(iterator, window):
            in_flight.append((item, pool.submit(fn, item)))
        while in_flight:
            item, future = in_flight.popleft()
            try:
                if task_timeout is not None:
                    result = future.result(timeout=task_timeout)
                else:
                    result = future.result()
                yield item, result, None
            except TimeoutError:
                # future.cancel() only prevents pending tasks from starting;
                # it cannot kill an already-running thread.  The abandoned
                # thread will eventually finish (or die with the process).
                future.cancel()
                yield item, None, TimeoutError(f"Task timed out after {task_timeout}s")
            except Exception as exc:
                yield item, None, exc
            for nxt in islice(iterator, 1):
                in_flight.append((nxt, pool.submit(fn, nxt)))
