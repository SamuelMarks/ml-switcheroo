"""Test module."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.passes.auxiliary import AuxiliaryTransformer
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig


def setup_ctx(alias_map=None):
  """Test function."""
  config = RuntimeConfig(source_framework="torch", target_framework="torch")
  ctx = RewriterContext(semantics=SemanticsManager(), config=config)
  if alias_map:
    ctx.alias_map = alias_map
  return ctx


def test_auxiliary_traits_empty():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={})
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None
  assert p._get_traits() is traits


def test_auxiliary_get_qualified_name_alias_split():
  """Test function."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Attribute(value=cst.Name("np"), attr=cst.Name("add"))
  assert p._get_qualified_name(node) == "numpy.add"


def test_auxiliary_get_qualified_name_no_string():
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = cst.Integer("1")
  assert p._get_qualified_name(node) is None


def test_auxiliary_create_dotted_name():
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  node = p._create_dotted_name("a.b.c")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "c"


def test_auxiliary_leave_simplestatementline_warnings():
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_warnings = ["warn1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node
  assert hasattr(res, "nodes")  # FlattenSentinel


def test_auxiliary_leave_simplestatementline_errors():
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  p.context.current_stmt_errors = ["err1"]
  node = cst.SimpleStatementLine(body=[cst.Pass()])
  res = p.leave_SimpleStatementLine(node, node)
  assert res is not node


def test_auxiliary_leave_decorator_remove():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": None}}))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert type(res).__name__ == "RemovalSentinel"


def test_auxiliary_leave_decorator_rename_noncall():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p = AuxiliaryTransformer(ctx)
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)
  assert isinstance(res.decorator, cst.Name)
  assert res.decorator.value == "new_dec"


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node, hook_ctx):
    """Mocks the hook."""
    if hook_ctx is ctx.hook_context:
      return cst.Pass()
    return node

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_exception(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node, hook_ctx):
    """Mocks the hook."""
    raise ValueError("static error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert res is loop


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node, hook_ctx):
    """Mocks the hook."""
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_hook_exception(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node, hook_ctx):
    """Mocks the hook."""
    raise ValueError("loop error")

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop" else None
  res = p.leave_For(loop, loop)
  assert not isinstance(res, cst.For)


def test_auxiliary_traits_with_traits():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_framework_config = MagicMock(return_value={"traits": {}})
  p = AuxiliaryTransformer(ctx)
  traits = p._get_traits()
  assert traits is not None


def test_auxiliary_get_qualified_name_alias_no_split():
  """Test function."""
  ctx = setup_ctx({"np": "numpy"})
  p = AuxiliaryTransformer(ctx)
  node = cst.Name("np")
  assert p._get_qualified_name(node) == "numpy"


def test_auxiliary_leave_decorator_rename_noncall2():
  """Test function."""
  ctx = setup_ctx()
  ctx.semantics.get_definition = MagicMock(return_value=("id", {"variants": {"torch": {"api": "new_dec"}}}))
  p = AuxiliaryTransformer(ctx)
  # The actual decorator decorator is just a Name, not a Call
  dec = cst.Decorator(decorator=cst.Name("test"))
  res = p.leave_Decorator(dec, dec)
  assert isinstance(res, cst.Decorator)


@patch("ml_switcheroo.core.rewriter.passes.auxiliary.get_hook")
def test_auxiliary_for_loop_static_hook_new_node(mock_get_hook):
  """Test function."""
  ctx = setup_ctx()
  p = AuxiliaryTransformer(ctx)
  loop = cst.For(
    target=cst.Name("i"),
    iter=cst.Name("range"),
    body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
  )

  def hook_mock(node, hook_ctx):
    """Mocks the hook."""
    return cst.Pass()

  mock_get_hook.side_effect = lambda name: hook_mock if name == "transform_for_loop_static" else None
  res = p.leave_For(loop, loop)
  assert isinstance(res, cst.Pass)
