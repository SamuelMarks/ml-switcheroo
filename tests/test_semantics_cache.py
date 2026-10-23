"""Test suite for semantics cache."""

import threading

from ml_switcheroo.semantics.cache import LRUCache


def test_lru_cache_basic() -> None:
  """Verify lru cache basic behavior."""
  cache: LRUCache[str, int] = LRUCache(capacity=2)
  cache["a"] = 1
  cache["b"] = 2
  assert "a" in cache
  assert cache["a"] == 1
  assert cache["b"] == 2

  # Exceed capacity
  cache["c"] = 3
  # "a" was least recently used since "b" was accessed last
  assert "a" not in cache
  assert "b" in cache
  assert "c" in cache

  assert cache.get("c") == 3
  assert cache.get("a", default=99) == 99


def test_lru_cache_update_existing() -> None:
  """Verify lru cache update existing behavior."""
  cache: LRUCache[str, int] = LRUCache(capacity=2)
  cache["a"] = 1
  cache["b"] = 2
  cache["a"] = 10

  cache["c"] = 3
  assert "a" in cache
  assert cache["a"] == 10
  assert "b" not in cache


def test_lru_cache_clear() -> None:
  """Verify lru cache clear behavior."""
  cache: LRUCache[str, int] = LRUCache(capacity=2)
  cache["a"] = 1
  cache.clear()
  assert "a" not in cache
  assert cache.get("a") is None


def test_lru_cache_thread_safety() -> None:
  """Verify lru cache thread safety behavior."""
  cache: LRUCache[int, int] = LRUCache(capacity=1000)

  def worker(start: int, end: int) -> None:
    """Execute worker test helper.

    Args:
        start: The start (int) for worker.
        end: The end (int) for worker.
    """
    for i in range(start, end):
      cache[i] = i
      _ = cache[i]
      _ = i in cache
      _ = cache.get(i)

  threads = []
  for i in range(10):
    t = threading.Thread(target=worker, args=(i * 100, (i + 1) * 100))
    threads.append(t)
    t.start()

  for t in threads:
    t.join()

  # The cache should have a valid state, capacity should not exceed 1000
  assert len(cache.cache) == 1000
