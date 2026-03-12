import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.hooks import HookContext, _HOOKS
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import FrameworkVariant

_sem = SemanticsManager()


def make_ctx(target="torch"):
  config = RuntimeConfig(source_framework="jax", target_framework=target)
  hctx = HookContext(
    semantics=_sem,
    config=config,
  )
  hctx.current_op_id = "Conv2d"
  return hctx


def get_ast_nodes():
  return [
    cst.Call(func=cst.Name("func")),
    cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("y"))),
    cst.Name("some_var"),
    cst.Attribute(value=cst.Name("obj"), attr=cst.Name("attr")),
    cst.Call(
      func=cst.Name("func"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("y"), keyword=cst.Name("kw"))]
    ),
    cst.Call(
      func=cst.Name("func"),
      args=[
        cst.Arg(value=cst.Integer("1")),
        cst.Arg(value=cst.List([cst.Element(cst.Integer("2"))]), keyword=cst.Name("padding")),
      ],
    ),
    cst.Call(
      func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("scatter")),
      args=[
        cst.Arg(value=cst.Integer("1"), keyword=cst.Name("index")),
        cst.Arg(value=cst.Integer("1"), keyword=cst.Name("value")),
      ],
    ),
    cst.Call(
      func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("flatten")),
      args=[cst.Arg(value=cst.Integer("1")), cst.Arg(value=cst.Integer("1"))],
    ),
    cst.For(
      target=cst.Name("i"),
      iter=cst.Name("iterable"),
      body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
    ),
    cst.FunctionDef(
      name=cst.Name("foo"),
      params=cst.Parameters(),
      body=cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
    ),
    cst.Assign(targets=[cst.AssignTarget(cst.Name("foo"))], value=cst.Call(func=cst.Name("some_func"))),
  ]


def test_plugin_coverage_fuzz():
  import ml_switcheroo.plugins

  # Fuzz all hooks with lots of configurations and variants
  nodes = get_ast_nodes()
  targets = ["torch", "jax", "mlx", "tensorflow", "keras", "source_placeholder"]

  contexts = {tgt: make_ctx(target=tgt) for tgt in targets}

  for hook_name, hook_func in _HOOKS.items():
    for node in nodes:
      for tgt, ctx in contexts.items():
        try:
          hook_func(node, ctx)
        except Exception:
          pass
