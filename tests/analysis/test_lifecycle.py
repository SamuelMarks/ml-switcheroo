"""Test suite for the Lifecycle module."""

import libcst as cst
from ml_switcheroo.analysis.lifecycle import InitializationTracker


def scan_code(code: str) -> list[str]:
  """Scans code."""
  wrapper = cst.parse_module(code)
  tracker = InitializationTracker()
  wrapper.visit(tracker)
  return tracker.warnings


def test_valid_init_usage():
  """Verifies the behavior of valid initialization usage."""
  code = "\nclass Model:\n    def __init__(self):\n        self.conv = 1\n    def forward(self, x):\n        return self.conv(x)\n"
  warnings = scan_code(code)
  assert not warnings


def test_missing_init():
  """Verifies the behavior of missing initialization."""
  code = "\nclass Model:\n    def __init__(self):\n        pass\n    def forward(self, x):\n        return self.conv(x)\n"
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "Members used in forward/call but not initialized" in warnings[0]
  assert "conv" in warnings[0]


def test_call_alias():
  """Verifies the behavior of call alias."""
  code = (
    "\nclass Model:\n    def __init__(self):\n        pass\n    def __call__(self, x):\n        return self.missing\n"
  )
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "missing" in warnings[0]


def test_multiple_missing():
  """Verifies the behavior of multiple missing."""
  code = "\nclass Model:\n    def __init__(self):\n        self.ok = 1\n    def forward(self):\n        return self.ok + self.missing1 + self.missing2\n"
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "missing1" in warnings[0]
  assert "missing2" in warnings[0]


def test_annotated_assignment():
  """Verifies the behavior of annotated assignment."""
  code = (
    "\nclass Model:\n    def __init__(self):\n        self.x: int = 1\n    def forward(self):\n        return self.x\n"
  )
  warnings = scan_code(code)
  assert not warnings


def test_tuple_unpacking_assignment():
  """Verifies the behavior of tuple unpacking assignment."""
  code = "\nclass Model:\n    def __init__(self):\n        self.x, self.y = 1, 2\n    def forward(self):\n        return self.x + self.y\n"
  warnings = scan_code(code)
  assert not warnings


def test_nested_classes():
  """Verifies the behavior of nested classes."""
  code = "\nclass Outer:\n    def __init__(self):\n        self.outer_val = 1\n\n    class Inner:\n        def __init__(self):\n            pass\n        def forward(self):\n            return self.inner_missing  # Missing in Inner\n\n    def forward(self):\n        return self.outer_val\n"
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "Class 'Inner'" in warnings[0]
  assert "inner_missing" in warnings[0]
  assert "outer_val" not in warnings[0]


def test_ignore_assignments_in_forward():
  """Verifies the behavior of ignore assignments in forward."""
  code = "\nclass Model:\n    def __init__(self):\n        pass\n    def forward(self):\n        self.dynamic = 1\n        return self.dynamic\n"
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "dynamic" in warnings[0]


def test_module_level_constructs():
  """Verifies the behavior of module level constructs."""
  code = "\ndef my_func():\n    pass\n\ndef __init__():\n    pass\n\nself.x = 1\nself.y: int = 2\n\nz = self.x\n"
  issues = scan_code(code)
  assert len(issues) == 0


def test_defensive_leave_classdef():
  """Verifies the behavior of defensive leave classdef."""
  tracker = InitializationTracker()
  tracker.leave_ClassDef(None)


def test_other_branches():
  """Verifies the behavior of other branches."""
  code = "\nclass Model:\n    def __init__(self):\n        x = 1\n        self.ok = 1\n\n    def other_method(self):\n        self.dynamic: int = 1\n\n    def forward(self, input_tensor):\n        shape = input_tensor.shape\n        return self.ok\n"
  warnings = scan_code(code)
  assert not warnings
