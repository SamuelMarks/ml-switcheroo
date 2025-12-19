"""
Plugins Package.

Exports standard plugins.
"""
# ... existing imports ...

# Register the Loop Unroller
from .static_unroll import unroll_static_loops
