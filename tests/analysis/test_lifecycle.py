"""Test module."""

import libcst as cst

from ml_switcheroo.analysis.lifecycle import InitializationTracker


def test_initialization_tracker_basic() -> None:
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
  tracker: InitializationTracker = InitializationTracker()
  tracker.leave_ClassDef(cst.ClassDef(name=cst.Name("Dummy"), body=cst.IndentedBlock(body=[])))
  assert len(tracker.warnings) == 0


# --- Merged from test_lifecycle_extra.py ---


def analyze(code: str) -> InitializationTracker:
  """Analyze code for initialization tracker."""
  tree: cst.Module = cst.parse_module(code)
  tracker: InitializationTracker = InitializationTracker()
  tree.visit(tracker)
  return tracker


def test_initialization_tracker_basic_extra() -> None:
  """Docstring."""
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
  """Docstring."""
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
  """Docstring."""
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
