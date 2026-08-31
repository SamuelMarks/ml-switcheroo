"""Test module for the InitializationTracker in lifecycle analysis.

This module verifies that the `InitializationTracker` correctly monitors variable
lifecycle within Neural Network module classes. It ensures that stateful attributes
accessed in `forward` passes or other methods are properly initialized in the `__init__`
constructor, issuing warnings for potential uninitialized variable access.
"""

import libcst as cst

from ml_switcheroo.analysis.lifecycle import InitializationTracker


def test_initialization_tracker_basic() -> None:
  """Test the correct tracking of basic attribute initializations.

  Verifies that when a class defines attributes in `__init__` (like `self.conv`)
  and uses them in `forward`, the tracker considers them safely initialized and
  emits no warnings.
  """
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
class MyModule:
    def __init__(self):
        self.conv = nn.Conv2d()
        self.bias = 0.1
        self.unused = True

    def forward(self, x):
        x = self.conv(x)
        return x + self.bias
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_missing() -> None:
  """Test detection of uninitialized class attributes.

  Verifies that if a variable (like `self.missing_bias`) is accessed in `forward`
  but was never assigned in `__init__`, the tracker successfully flags it and
  generates a warning.
  """
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
class BadModule:
    def __init__(self):
        self.conv = nn.Conv2d()

    def forward(self, x):
        x = self.conv(x)
        return x + self.missing_bias
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 1
  assert "BadModule" in tracker.warnings[0]
  assert "missing_bias" in tracker.warnings[0]


def test_initialization_tracker_tuple_unpacking() -> None:
  """Test initialization tracking through tuple and list unpacking.

  Ensures that the tracker can correctly parse multiple assignments like
  `(self.a, self.b) = (1, 2)` or `[self.c, self.d] = [3, 4]` within `__init__`.
  """
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
class TupleMod:
    def __init__(self):
        (self.a, self.b) = (1, 2)
        [self.c, self.d] = [3, 4]

    def forward(self, x):
        return x + self.a + self.b + self.c + self.d
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_annassign() -> None:
  """Test initialization tracking for annotated assignments.

  Verifies that type-annotated assignments in `__init__` (e.g. `self.a: int = 1`)
  are successfully tracked as valid initializations.
  """
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
class AnnMod:
    def __init__(self):
        self.a: int = 1

    def forward(self, x):
        return x + self.a
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_nested() -> None:
  """Test tracking within nested class definitions.

  Verifies that the tracker manages scope stacks correctly for nested classes,
  ensuring that missing initializations are flagged for the correct inner
  or outer class scope.
  """
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
class Outer:
    def __init__(self):
        self.outer_var = 1

    class Inner:
        def __init__(self):
            self.inner_var = 2
        def forward(self, x):
            return x + self.inner_var + self.missing_inner

    def forward(self, x):
        return x + self.outer_var + self.missing_outer
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 2
  assert any("Inner" in w and "missing_inner" in w for w in tracker.warnings)
  assert any("Outer" in w and "missing_outer" in w for w in tracker.warnings)


def test_initialization_tracker_no_scope() -> None:
  """Test tracker resilience outside of class contexts.

  Verifies that if `__init__` or `forward` style functions are defined globally
  (outside a class scope), the tracker safely ignores them without crashing
  or emitting false warnings.
  """
  # Test methods returning early when scope stack is empty (e.g. methods outside classes)
  tracker: InitializationTracker = InitializationTracker()

  code: str = """
def __init__(self):
    self.x = 1
    self.y: int = 2
    (self.z, self.w) = (3, 4)

def forward(self, x):
    return self.x + self.missing
"""
  tree: cst.Module = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_leave_classdef_no_scope() -> None:
  """Test defensive programming in `leave_ClassDef` when state is missing.

  Ensures the tracker handles scenarios where `leave_ClassDef` is called
  but the internal scope stack has somehow been emptied prematurely.
  """
  tracker: InitializationTracker = InitializationTracker()
  tracker.leave_ClassDef(cst.ClassDef(name=cst.Name("Dummy"), body=cst.IndentedBlock(body=[])))
  assert len(tracker.warnings) == 0


# --- Merged from test_lifecycle_extra.py ---


def analyze(code: str) -> InitializationTracker:
  """Helper to run the InitializationTracker on a snippet of code.

  Args:
      code (str): Source code to parse and analyze.

  Returns:
      InitializationTracker: The populated tracker instance after traversal.
  """
  tree: cst.Module = cst.parse_module(code)
  tracker: InitializationTracker = InitializationTracker()
  tree.visit(tracker)
  return tracker


def test_initialization_tracker_basic_extra() -> None:
  """Test basic attribute initialization using the analyzer helper."""
  code: str = """
class MyModule:
    def __init__(self):
        self.w = 1.0

    def forward(self, x):
        return self.w * x
    """
  tracker: InitializationTracker = analyze(code)
  assert len(tracker.warnings) == 0


def test_initialization_tracker_uninitialized() -> None:
  """Test detection of variables accessed but completely uninitialized in `__init__`."""
  code: str = """
class MyModule:
    def __init__(self):
        pass

    def forward(self, x):
        return self.w * x
    """
  tracker: InitializationTracker = analyze(code)
  assert len(tracker.warnings) == 1


def test_initialization_tracker_complex() -> None:
  """Test complex assignment issues, including intra-initialization and late initialization.

  Verifies that assigning an uninitialized variable to another variable *inside*
  `__init__` (e.g. `self.b = self.c`), or late-initializing variables inside
  `forward` (e.g. `self.d = 4`) correctly trigger warnings.
  """
  code: str = """
class SubModule:
    def __init__(self):
        self.a = 2

class MyModule:
    def __init__(self):
        self.sub = SubModule()
        self.b = self.c # c uninitialized

    def forward(self, x):
        self.d = 4 # late init
        return self.sub.a * self.w * x + self.d
    """
  tracker: InitializationTracker = analyze(code)
  assert len(tracker.warnings) > 0


def test_initialization_tracker_missing_branches() -> None:
  """Test branch coverage gaps for ignored statement types.

  Ensures that local variable assignments and non-init method definitions
  are safely bypassed by the tracker without causing analysis faults.
  """
  tracker = InitializationTracker()

  code = """
class MiscMod:
    def __init__(self):
        # 196->exit: Target is not Attribute on self or tuple/list
        local_var = 1

    def helper(self):
        # 103->exit and 123->exit: func_name not in init or forward
        # 153->exit: AnnAssign but not in init
        self.other: int = 1
        pass
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0
