"""Type definitions for the Sphinx extension.

Defines common data structures used across the registry and rendering modules
to ensure type consistency.
"""

import typing

#: Map of Parent Key -> List of {key, label} for children.
HierarchyMap = typing.Dict[str, typing.List[typing.Dict[str, str]]]
