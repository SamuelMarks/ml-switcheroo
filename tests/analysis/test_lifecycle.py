"""
Tests for InitializationTracker.
"""

import libcst as cst
from ml_switcheroo.analysis.lifecycle import InitializationTracker


def scan_code(code: str) -> list[str]:
  wrapper = cst.parse_module(code)
  tracker = InitializationTracker()
  wrapper.visit(tracker)
  return tracker.warnings


def test_valid_init_usage():
  code = """ 
class Model: 
    def __init__(self): 
        self.conv = 1
    def forward(self, x): 
        return self.conv(x) 
"""
  warnings = scan_code(code)
  assert not warnings


def test_missing_init():
  code = """ 
class Model: 
    def __init__(self): 
        pass
    def forward(self, x): 
        return self.conv(x) 
"""
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "Members used in forward/call but not initialized" in warnings[0]
  assert "conv" in warnings[0]


def test_call_alias():
  code = """ 
class Model: 
    def __init__(self): 
        pass
    def __call__(self, x): 
        return self.missing
"""
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "missing" in warnings[0]


def test_multiple_missing():
  code = """ 
class Model: 
    def __init__(self): 
        self.ok = 1
    def forward(self): 
        return self.ok + self.missing1 + self.missing2
"""
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "missing1" in warnings[0]
  assert "missing2" in warnings[0]


def test_annotated_assignment():
  code = """ 
class Model: 
    def __init__(self): 
        self.x: int = 1
    def forward(self): 
        return self.x
"""
  warnings = scan_code(code)
  assert not warnings


def test_tuple_unpacking_assignment():
  code = """ 
class Model: 
    def __init__(self): 
        self.x, self.y = 1, 2
    def forward(self): 
        return self.x + self.y
"""
  warnings = scan_code(code)
  assert not warnings


def test_nested_classes():
  code = """ 
class Outer: 
    def __init__(self): 
        self.outer_val = 1
    
    class Inner: 
        def __init__(self): 
            pass
        def forward(self): 
            return self.inner_missing  # Missing in Inner

    def forward(self): 
        return self.outer_val
"""
  warnings = scan_code(code)
  assert len(warnings) == 1
  assert "Class 'Inner'" in warnings[0]
  assert "inner_missing" in warnings[0]
  # Use distinct name 'outer_val' because prose contains 'a', 'b', etc.
  assert "outer_val" not in warnings[0]


def test_ignore_assignments_in_forward():
  """
  Even if assigned in forward, it should technically be in init for static guarantees.
  But specifically, we track assignments in __init__.
  If assigned in forward AND used in forward, it's missed by __init__ scan,
  so it should warn (as dynamic definition).
  """
  code = """ 
class Model: 
    def __init__(self): 
        pass
    def forward(self): 
        self.dynamic = 1
        return self.dynamic
"""
  warnings = scan_code(code)
  # self.dynamic is used, but not found in __init__. Should warn.
  assert len(warnings) == 1
  assert "dynamic" in warnings[0]
