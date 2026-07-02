"""Tests for the bounded, order-preserving parallel map."""

import threading
import time

from wireless_taxonomy.parallel import parallel_map


def test_sequential_fallback_preserves_order():
    items = list(range(10))
    out = list(parallel_map(lambda x: x * 2, items, workers=1))
    assert [item for item, _, _ in out] == items
    assert [result for _, result, _ in out] == [x * 2 for x in items]
    assert all(error is None for _, _, error in out)


def test_parallel_preserves_input_order():
    items = list(range(50))

    def slow_reverse_priority(x):
        # Later items finish first — output order must still match input.
        time.sleep((50 - x) * 0.001)
        return x * 2

    out = list(parallel_map(slow_reverse_priority, items, workers=8))
    assert [item for item, _, _ in out] == items
    assert [result for _, result, _ in out] == [x * 2 for x in items]


def test_errors_are_yielded_not_raised():
    def maybe_fail(x):
        if x % 3 == 0:
            raise ValueError(f"boom {x}")
        return x

    out = list(parallel_map(maybe_fail, range(10), workers=4))
    for item, result, error in out:
        if item % 3 == 0:
            assert result is None
            assert isinstance(error, ValueError)
        else:
            assert result == item
            assert error is None


def test_actually_runs_concurrently():
    active = 0
    peak = 0
    lock = threading.Lock()

    def track(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return x

    list(parallel_map(track, range(20), workers=4))
    assert peak > 1  # concurrency actually happened
    assert peak <= 4  # bounded by worker count


def test_lazy_item_consumption_bounded_window():
    """Items must be pulled lazily so memory stays bounded."""
    pulled = []

    def items_gen():
        for i in range(100):
            pulled.append(i)
            yield i

    gen = parallel_map(lambda x: x, items_gen(), workers=2, window=4)
    # Pull only the first result; the generator should not have consumed
    # everything up front.
    next(gen)
    assert len(pulled) < 100
    gen.close()


def test_workers_one_runs_inline_no_threads():
    thread_ids = set()

    def record(x):
        thread_ids.add(threading.get_ident())
        return x

    list(parallel_map(record, range(5), workers=1))
    assert thread_ids == {threading.get_ident()}
