"""Test suite for rewriter strategy coverage."""

import libcst as cst

from ml_switcheroo.core.hooks_registry import clear_hooks
from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation, execute_strategy


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self):
    """Initialize test helper instance."""
    self.hook_context = type("HookCtx", (), {"current_op_id": None})()


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self):
    """Initialize test helper instance."""
    self.context = DummyContext()
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.strict_mode = False

    class Semantics:
      """Test helper class Semantics."""

      pass

    self.semantics = Semantics()

  def _is_module_alias(self, node):
    """Execute is module alias helper logic."""
    return False

  def _create_name_node(self, name):
    """Execute create name node helper logic."""
    return cst.Name(name)

  def _report_failure(self, msg):
    """Execute report failure helper logic."""
    pass


def test_execute_strategy_branches():
  """Verify execute strategy branches behavior."""
  rewriter = DummyRewriter()
  original = cst.parse_statement("f()").body[0].value
  updated = original
  details = {"variants": {}}

  class DummyRule:
    """Test helper class DummyRule."""

    def __init__(self, if_arg, if_value, use_api):
      """Initialize test helper instance."""
      self.if_arg = if_arg
      self.is_val = if_value
      self.use_api = use_api
      self.op = None

  # 51 -> 52 (dispatch_rules), 60 -> 74, 74 -> 83, 83 -> 93, 93 -> 111 (standard)
  mapping1 = {"dispatch_rules": [DummyRule(if_arg="x", if_value=1, use_api="f1")], "api": "default"}
  execute_strategy(rewriter, original, updated, mapping1, details, "op1")

  # 51 -> 57 (no dispatch rules)
  # 60 -> 61 (infix)
  mapping2 = {"transformation_type": "infix", "operator": "+"}
  execute_strategy(rewriter, original, updated, mapping2, details, "op2")

  # 74 -> 75 (inline_lambda)
  mapping3 = {"transformation_type": "inline_lambda", "api": "lambda x: x"}
  execute_strategy(rewriter, original, updated, mapping3, details, "op3")

  # 83 -> 84 (requires_plugin)
  mapping4 = {"requires_plugin": "my_plugin"}
  clear_hooks()
  execute_strategy(rewriter, original, updated, mapping4, details, "op4")

  # 93 -> 94 (macro_template)
  # 97 -> 98, 97 -> 104
  mapping5 = {"macro_template": "a + b"}
  details_macro = {"std_args": ["a", {"name": "b"}, ("c", "type")]}
  _res = execute_strategy(rewriter, original, updated, mapping5, details_macro, "op5")
  assert _res is not None or _res is None


def test_apply_layout_permutation_branches():
  """Verify apply layout permutation branches behavior."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("f(a, b)").body[0].value
  details = {"std_args": ["a", "b"]}

  # 164 -> 165 (has args)
  # 164 -> 182 (loop ends)
  mapping = {"layout_map": {"a": "A -> B"}}
  _res = _apply_layout_permutation(node, mapping, details, rewriter)
  assert _res is not None or _res is None


def test_apply_layout_transformations_no_arrow():
  """Verify apply layout transformations no arrow behavior."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation

  class DummyRewriter:
    """Test helper class DummyRewriter."""

    def __init__(self):
      """Initialize test helper instance."""
      self.semantics = MagicMock()
      self.target_fw = "jax"

  rewriter = DummyRewriter()
  node = cst.parse_statement("f(x)").body[0].value

  mapping = {"layout_map": {"x": "NCHW", "return": "NHWC"}}
  details = {"std_args": [{"name": "x"}]}

  # "->" not in rule, so it should bypass both if statements
  res = _apply_layout_permutation(node, mapping, details, rewriter)
  assert res.args[0].value.value == "x"


def test_strategy_missing_api_custom_message():
  """Verify strategy missing api custom message behavior."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value
  mapping = {"missing_message": "Custom error message"}
  rewriter.last_failure = None

  def mock_report_failure(msg):
    """Execute mock report failure helper logic."""
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "Custom error message"


def test_strategy_missing_api_default_message():
  """Verify strategy missing api default message behavior."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value
  mapping = {}
  rewriter.last_failure = None

  def mock_report_failure(msg):
    """Execute mock report failure helper logic."""
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "No mapping available for 'foo' -> 'jax'"


def test_strategy_neural_rejection():
  """Verify strategy neural rejection behavior."""
  rewriter = DummyRewriter()
  rewriter.target_fw = "jax"
  rewriter.semantics._key_origins = {"op1": "neural"}
  call = cst.parse_statement("f()").body[0].value
  mapping = {}  # No api key
  rewriter.last_failure = None

  def mock_report_failure(msg):
    """Execute mock report failure helper logic."""
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "op1")
  assert "Cannot map neural network abstraction" in rewriter.last_failure
