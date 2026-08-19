"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy, _apply_layout_permutation
from ml_switcheroo.core.hooks_registry import clear_hooks


class DummyContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.hook_context = type("HookCtx", (), {"current_op_id": None})()


class DummyRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.context = DummyContext()
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.strict_mode = False

    class Semantics:
      """Class doc."""

      pass

    self.semantics = Semantics()

  def _is_module_alias(self, node):
    """Function doc."""
    return False

  def _create_name_node(self, name):
    """Function doc."""
    return cst.Name(name)

  def _report_failure(self, msg):
    """Function doc."""
    pass


def test_execute_strategy_branches():
  """Docstring."""
  rewriter = DummyRewriter()
  original = cst.parse_statement("f()").body[0].value
  updated = original
  details = {"variants": {}}

  class DummyRule:
    """Class doc."""

    def __init__(self, if_arg, if_value, use_api):
      """Init doc."""
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
  execute_strategy(rewriter, original, updated, mapping5, details_macro, "op5")


def test_apply_layout_permutation_branches():
  """Docstring."""
  rewriter = DummyRewriter()
  node = cst.parse_statement("f(a, b)").body[0].value
  details = {"std_args": ["a", "b"]}

  # 164 -> 165 (has args)
  # 164 -> 182 (loop ends)
  mapping = {"layout_map": {"a": "A -> B"}}
  _apply_layout_permutation(node, mapping, details, rewriter)


def test_apply_layout_transformations_no_arrow():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.calls.strategy import _apply_layout_permutation
  from unittest.mock import MagicMock

  class DummyRewriter:
    """Class doc."""

    def __init__(self):
      """Init doc."""
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
  """Docstring."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value
  mapping = {"missing_message": "Custom error message"}
  rewriter.last_failure = None

  def mock_report_failure(msg):
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "Custom error message"


def test_strategy_missing_api_default_message():
  """Docstring."""
  rewriter = DummyRewriter()
  call = cst.parse_statement("foo()").body[0].value
  mapping = {}
  rewriter.last_failure = None

  def mock_report_failure(msg):
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "foo")
  assert rewriter.last_failure == "No mapping available for 'foo' -> 'jax'"


def test_strategy_neural_rejection():
  """Docstring."""
  rewriter = DummyRewriter()
  rewriter.target_fw = "jax"
  rewriter.semantics._key_origins = {"op1": "neural"}
  call = cst.parse_statement("f()").body[0].value
  mapping = {}  # No api key
  rewriter.last_failure = None

  def mock_report_failure(msg):
    rewriter.last_failure = msg

  rewriter._report_failure = mock_report_failure
  execute_strategy(rewriter, call, call, mapping, {}, "op1")
  assert "Cannot map neural network abstraction" in rewriter.last_failure
