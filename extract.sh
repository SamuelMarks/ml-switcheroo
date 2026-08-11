#!/bin/bash
head -n 1007 src/ml_switcheroo/core/compiler/backends/sass/macros.py > src/ml_switcheroo/core/compiler/backends/sass/macros_core.py

cat << 'INNER_EOF' > src/ml_switcheroo/core/compiler/backends/sass/macros_extra.py
"""SASS Macro Expansion Logic - Extra Macros."""
from typing import List, Dict, Any, Callable
from ml_switcheroo.core.compiler.frontends.sass.cst import (
  SassOperand, SassInstruction, SassLabel, SassRegister,
  SassImmediate, SassMemory, SassPredicate, SassComment, SassNode,
)
from .macros_core import RegisterAllocatorProtocol

INNER_EOF

tail -n +1008 src/ml_switcheroo/core/compiler/backends/sass/macros.py >> src/ml_switcheroo/core/compiler/backends/sass/macros_extra.py

cat << 'INNER_EOF2' >> src/ml_switcheroo/core/compiler/backends/sass/macros_core.py

from .macros_extra import *
INNER_EOF2

mv src/ml_switcheroo/core/compiler/backends/sass/macros_core.py src/ml_switcheroo/core/compiler/backends/sass/macros.py
