"""
Tests for Container Type Inference in Harvester.

Verifies:
1. Tuple literals `(1, 2)` map to `Tuple[int]` specs.
2. List literals `[1, 2]` map to `List[int]` specs.
3. Nested containers `[(1, 1)]` map to `List[Tuple[int]]`.
4. Empty containers fall back to generic types.
"""

import pytest
import ast
from ml_switcheroo.discovery.harvester import TargetCallVisitor


@pytest.fixture
def visitor_factory():
  """Factory to create visitor with custom std_args."""

  def create(std_args):
    return TargetCallVisitor("target.api", {}, std_args)

  return create


def test_tuple_int_inference(visitor_factory):
  """
  Scenario: `target.api(dims=(0, 2))`
  Spec: `axes: Tuple[int]`
  Expect: `dims` -> `axes`
  """
  std_args = [("x", "Tensor"), ("axes", "Tuple[int, ...]")]
  visitor = visitor_factory(std_args)

  # Construct AST call
  # call(dims=(0, 2))
  call = ast.Call(
    func=ast.Name(id="target"),
    args=[],
    keywords=[
      ast.keyword(arg="dims", value=ast.Tuple(elts=[ast.Constant(value=0), ast.Constant(value=2)], ctx=ast.Load()))
    ],
  )

  # Manually trigger mapping logic (normally done via visit)
  visitor._resolve_call_name = lambda x: "target.api"
  visitor.visit(call)

  # Expect explicit type inference to map 'dims' value -> 'axes' standard arg.
  assert visitor.mappings is not None
  assert visitor.mappings.get("axes") == "dims"


def test_list_int_inference(visitor_factory):
  """
  Scenario: `target.api(pad=[1, 1])`
  Spec: `padding: List[int]`
  Expect: `pad` -> `padding`
  """
  std_args = [("x", "Tensor"), ("padding", "List[int]")]
  visitor = visitor_factory(std_args)

  visitor._resolve_call_name = lambda x: "target.api"

  call = ast.Call(
    func=ast.Name(id="target"),
    args=[],
    keywords=[
      ast.keyword(arg="pad", value=ast.List(elts=[ast.Constant(value=1), ast.Constant(value=1)], ctx=ast.Load()))
    ],
  )
  visitor.visit(call)

  assert visitor.mappings.get("padding") == "pad"


def test_nested_container_inference(visitor_factory):
  """
  Scenario: `target.api(widths=[(1, 2), (3, 4)])`
  Spec: `config: List[Tuple[int]]`
  """
  std_args = [("config", "List[Tuple[int]]")]
  visitor = visitor_factory(std_args)
  visitor._resolve_call_name = lambda x: "target.api"

  # [(1,), (2,)]
  inner1 = ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load())
  inner2 = ast.Tuple(elts=[ast.Constant(value=2)], ctx=ast.Load())
  outer = ast.List(elts=[inner1, inner2], ctx=ast.Load())

  call = ast.Call(func=ast.Name(id="target"), args=[], keywords=[ast.keyword(arg="widths", value=outer)])
  visitor.visit(call)

  assert visitor.mappings.get("config") == "widths"


def test_empty_container_fallback(visitor_factory):
  """
  Scenario: `target.api(opt=[])`
  Spec: `options: List[str]`
  Logic: `[]` infers as `List`. `List` in `List[str]`. Match.
  """
  std_args = [("options", "List[str]")]
  visitor = visitor_factory(std_args)
  visitor._resolve_call_name = lambda x: "target.api"

  call = ast.Call(
    func=ast.Name(id="target"), args=[], keywords=[ast.keyword(arg="opt", value=ast.List(elts=[], ctx=ast.Load()))]
  )
  visitor.visit(call)

  assert visitor.mappings.get("options") == "opt"


def test_mixed_types_generic_fallback(visitor_factory):
  """
  Scenario: `target.api(mix=[1, "a"])` (Mixed int/str)
  Result: Infers "List" (generic).
  Spec: `data: List` (generic).
  """
  std_args = [("data", "List")]
  visitor = visitor_factory(std_args)
  visitor._resolve_call_name = lambda x: "target.api"

  call = ast.Call(
    func=ast.Name(id="target"),
    args=[],
    keywords=[
      ast.keyword(arg="mix", value=ast.List(elts=[ast.Constant(value=1), ast.Constant(value="a")], ctx=ast.Load()))
    ],
  )
  visitor.visit(call)

  assert visitor.mappings.get("data") == "mix"
