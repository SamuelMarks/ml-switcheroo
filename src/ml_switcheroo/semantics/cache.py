"""Thread-safe bounded caching utilities for SemanticsManager."""

import typing
import threading
from collections import OrderedDict

K = typing.TypeVar("K")
V = typing.TypeVar("V")


class LRUCache(typing.Generic[K, V]):
  """Thread-safe LRU cache to prevent unbounded memory retention."""

  def __init__(self, capacity: int = 1000) -> None:
    """Initialize the bounded cache.

    Args:
        capacity: Maximum number of elements to store.
    """
    self.capacity = capacity
    self.cache: OrderedDict[K, V] = OrderedDict()
    self.lock = threading.Lock()

  def __contains__(self, key: K) -> bool:
    """Check if key is in cache.

    Args:
        key: Cache key to check.

    Returns:
        True if key exists in cache, False otherwise.
    """
    with self.lock:
      return key in self.cache

  def __getitem__(self, key: K) -> V:
    """Get an item from cache, marking it as recently used.

    Args:
        key: Cache key to retrieve.

    Returns:
        Cached value associated with ``key``.
    """
    with self.lock:
      value = self.cache.pop(key)
      self.cache[key] = value
      return value

  def __setitem__(self, key: K, value: V) -> None:
    """Set an item in cache, evicting the oldest if at capacity.

    Args:
        key: Cache key to store.
        value: Value to associate with ``key``.
    """
    with self.lock:
      if key in self.cache:
        self.cache.pop(key)
      self.cache[key] = value
      if len(self.cache) > self.capacity:
        self.cache.popitem(last=False)

  def get(self, key: K, default: V | None = None) -> V | None:
    """Get an item with a fallback default.

    Args:
        key: Cache key to look up.
        default: Fallback value if ``key`` is not present.

    Returns:
        Cached value if found, else ``default``.
    """
    with self.lock:
      if key in self.cache:
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
      return default

  def clear(self) -> None:
    """Clear all entries from the cache."""
    with self.lock:
      self.cache.clear()
