"""Built-in transformation plugins package for ml-switcheroo.

Plugin modules inside this package are discovered and loaded lazily on demand via
``HookRegistry.ensure_loaded()`` without mutating global state at package import time.
"""

__all__: list[str] = []
