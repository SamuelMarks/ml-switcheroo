"""Test suite for lifecycle analysis extra coverage."""

import libcst as cst
from ml_switcheroo.analysis.lifecycle import InitializationTracker


def analyze(code: str) -> InitializationTracker:
  """Analyze code for initialization tracker."""
  tree = cst.parse_module(code)
  tracker = InitializationTracker()
  tree.visit(tracker)
  return tracker


def test_initialization_tracker_basic():
  """Test basic initialization tracker behavior."""
  code = """
class MyModule:
    def __init__(self):
        self.w = 1.0

    def forward(self, x):
        return self.w * x
    """
  tracker = analyze(code)
  assert len(tracker.warnings) == 0


def test_initialization_tracker_uninitialized():
  """Test initialization tracker with uninitialized variables."""
  code = """
class MyModule:
    def __init__(self):
        pass

    def forward(self, x):
        return self.w * x
    """
  tracker = analyze(code)
  assert len(tracker.warnings) == 1


def test_initialization_tracker_complex():
  """Test initialization tracker with complex code."""
  code = """
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
  tracker = analyze(code)
  assert len(tracker.warnings) > 0
