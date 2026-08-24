"""Test module."""

import libcst as cst
from ml_switcheroo.analysis.lifecycle import InitializationTracker


def test_initialization_tracker_basic():
  """Test element."""
  tracker = InitializationTracker()

  code = """
class MyModule:
    def __init__(self):
        self.conv = nn.Conv2d()
        self.bias = 0.1
        self.unused = True

    def forward(self, x):
        x = self.conv(x)
        return x + self.bias
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_missing():
  """Test element."""
  tracker = InitializationTracker()

  code = """
class BadModule:
    def __init__(self):
        self.conv = nn.Conv2d()

    def forward(self, x):
        x = self.conv(x)
        return x + self.missing_bias
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 1
  assert "BadModule" in tracker.warnings[0]
  assert "missing_bias" in tracker.warnings[0]


def test_initialization_tracker_tuple_unpacking():
  """Test element."""
  tracker = InitializationTracker()

  code = """
class TupleMod:
    def __init__(self):
        (self.a, self.b) = (1, 2)
        [self.c, self.d] = [3, 4]

    def forward(self, x):
        return x + self.a + self.b + self.c + self.d
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_annassign():
  """Test element."""
  tracker = InitializationTracker()

  code = """
class AnnMod:
    def __init__(self):
        self.a: int = 1

    def forward(self, x):
        return x + self.a
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_nested():
  """Test element."""
  tracker = InitializationTracker()

  code = """
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
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 2
  assert any("Inner" in w and "missing_inner" in w for w in tracker.warnings)
  assert any("Outer" in w and "missing_outer" in w for w in tracker.warnings)


def test_initialization_tracker_no_scope():
  """Test element."""
  # Test methods returning early when scope stack is empty (e.g. methods outside classes)
  tracker = InitializationTracker()

  code = """
def __init__(self):
    self.x = 1
    self.y: int = 2
    (self.z, self.w) = (3, 4)

def forward(self, x):
    return self.x + self.missing
"""
  tree = cst.parse_module(code)
  tree.visit(tracker)

  assert len(tracker.warnings) == 0


def test_initialization_tracker_leave_classdef_no_scope():
  """Test element."""
  tracker = InitializationTracker()
  tracker.leave_ClassDef(cst.ClassDef(name=cst.Name("Dummy"), body=cst.IndentedBlock(body=[])))
  assert len(tracker.warnings) == 0
