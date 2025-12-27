"""
Type definitions for the Sphinx extension.

Defines common data structures used across the registry and rendering modules
to ensure type consistency.
"""

from typing import Dict, List

#: Map of Parent Key -> List of {key, label} for children.
HierarchyMap = Dict[str, List[Dict[str, str]]]
