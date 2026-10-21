"""Type definitions for the Sphinx extension.

Defines common data structures used across the registry and rendering modules
to ensure type consistency.
"""

#: Map of Parent Key -> List of {key, label} for children.
HierarchyMap = dict[str, list[dict[str, str]]]
