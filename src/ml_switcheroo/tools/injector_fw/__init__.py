"""
LibCST Transformer for Injecting Framework Mappings.

This package provides the logic to modify framework adapter files (e.g. `torch.py`)
by locating the specific class registered for a framework and injecting a new
`StandardMap` definition into its `definitions` property.

It handles:

1.  **Definitions Injection**: Appending the mapping to the definitions dictionary.
2.  **Smart Import Injection**: Analyzing the target API path (e.g. `scipy.special.erf`)
    and injecting necessary top-level imports (`import scipy`) if missing.
3.  **Variant Parameter Injection**: Supporting `inject_args` for adding fixed arguments.
4.  **Complex Literal Support**: Recursively converting Lists, Tuples, and Dicts to CST nodes.
"""

from ml_switcheroo.tools.injector_fw.core import FrameworkInjector
from ml_switcheroo.tools.injector_fw.utils import convert_to_cst_literal

# Internal alias to maintain compatibility with tests expecting the private name
_convert_to_cst_literal = convert_to_cst_literal

__all__ = ["FrameworkInjector", "_convert_to_cst_literal", "convert_to_cst_literal"]
