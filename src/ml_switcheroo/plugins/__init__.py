"""Plugin Package.

Automatically discovers and registers all plugin modules within this package
by delegating to ``ml_switcheroo.core.hooks_registry.load_plugins``.
"""

from ml_switcheroo.core.hooks_registry import load_plugins

load_plugins()
