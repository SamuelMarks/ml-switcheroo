"""Static Safety Scanner and Feasibility Verifier.

This module inspects Concrete Syntax Trees (CST) to identify constructs that
cannot be statically analyzed or compiled across the static conversion matrix.
These include value-dependent control flow, dynamic tensor boolean masking,
unbounded while loops, and dynamic Autograd invocations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import libcst as cst


class StaticSafetyCategory(str, Enum):
  """Categories of static analysis boundaries and violations."""

  DYNAMIC_SHAPE = "DYNAMIC_SHAPE"
  VALUE_DEPENDENT_CONTROL_FLOW = "VALUE_DEPENDENT_CONTROL_FLOW"
  DATA_DEPENDENT_LOOP = "DATA_DEPENDENT_LOOP"
  DYNAMIC_AUTOGRAD = "DYNAMIC_AUTOGRAD"


@dataclass(frozen=True)
class StaticSafetyDiagnostic:
  """Diagnostic information for a static safety violation.

  Attributes:
      category: The classification of the static violation.
      message: Explanation of the static analysis limitation.
      recommendation: Actionable refactoring guidance for static compatibility.
      snippet: Code snippet or representation of the offending construct.
  """

  category: StaticSafetyCategory
  message: str
  recommendation: str
  snippet: str


class StaticSafetyScanner(cst.CSTVisitor):
  """Visitor that flags dynamic and non-static language constructs in CST."""

  def __init__(self) -> None:
    """Initialize the StaticSafetyScanner with an empty diagnostics list."""
    self.diagnostics: List[StaticSafetyDiagnostic] = []

  def visit_Subscript(self, node: cst.Subscript) -> None:
    """Detect dynamic boolean tensor masking in subscripts.

    Args:
        node: The CST Subscript node being visited.
    """
    for slice_element in node.slice:
      slice_val = slice_element.slice
      if isinstance(slice_val, cst.Index):
        val = slice_val.value
        if isinstance(val, (cst.Comparison, cst.BooleanOperation, cst.BinaryOperation)):
          self.diagnostics.append(
            StaticSafetyDiagnostic(
              category=StaticSafetyCategory.DYNAMIC_SHAPE,
              message="Dynamic boolean tensor masking creates value-dependent output dimensions.",
              recommendation="Use static masking with where() or pad to fixed capacity with attention masks.",
              snippet="tensor[condition]",
            )
          )

  def visit_If(self, node: cst.If) -> None:
    """Detect value-dependent branching on tensor contents.

    Args:
        node: The CST If statement node being visited.
    """
    test_node = node.test
    is_tensor_value_branch = False

    class _ValueCallFinder(cst.CSTVisitor):
      """Nested visitor to locate tensor value extraction calls."""

      def __init__(self) -> None:
        """Initialize the finder."""
        self.found = False

      def visit_Call(self, call_node: cst.Call) -> None:
        """Inspect calls for tensor item() or evaluation methods.

        Args:
            call_node: Call node to inspect.
        """
        if isinstance(call_node.func, cst.Attribute):
          attr_name = call_node.func.attr.value
          if attr_name in ("item", "all", "any", "nonzero"):
            self.found = True

    finder = _ValueCallFinder()
    test_node.visit(finder)
    if finder.found:
      is_tensor_value_branch = True

    if is_tensor_value_branch:
      self.diagnostics.append(
        StaticSafetyDiagnostic(
          category=StaticSafetyCategory.VALUE_DEPENDENT_CONTROL_FLOW,
          message="Value-dependent branching on tensor values cannot be traced statically.",
          recommendation="Replace dynamic if/else branching with tensor selection (e.g. where/select).",
          snippet="if tensor.item() > ...:",
        )
      )

  def visit_While(self, node: cst.While) -> None:
    """Detect while loops without static trip counts.

    Args:
        node: The CST While statement node being visited.
    """
    # Unless the while loop test is a boolean literal True or False with a break,
    # flag data-dependent loops.
    is_static_counter = False
    if isinstance(node.test, cst.Comparison):
      # Check if right or left is a literal and comparator is comparison
      if isinstance(node.test.left, cst.Integer) or any(
        isinstance(comp.comparator, cst.Integer) for comp in node.test.comparisons
      ):
        is_static_counter = True

    if not is_static_counter:
      self.diagnostics.append(
        StaticSafetyDiagnostic(
          category=StaticSafetyCategory.DATA_DEPENDENT_LOOP,
          message="Dynamic while-loop without static trip count detected.",
          recommendation="Replace with bounded for-loop or functional scan operator (e.g., lax.scan).",
          snippet="while condition:",
        )
      )

  def visit_Call(self, node: cst.Call) -> None:
    """Detect dynamic Autograd and gradient invocations in forward paths.

    Args:
        node: The CST Call node being visited.
    """
    func_name: Optional[str] = None
    if isinstance(node.func, cst.Attribute):
      val = node.func.value
      if isinstance(val, cst.Attribute) and val.attr.value == "autograd":
        func_name = f"autograd.{node.func.attr.value}"
      elif isinstance(val, cst.Name) and val.value == "autograd":
        func_name = f"autograd.{node.func.attr.value}"
      elif node.func.attr.value in ("backward", "grad"):
        func_name = node.func.attr.value

    if func_name in ("autograd.grad", "autograd.backward", "backward"):
      self.diagnostics.append(
        StaticSafetyDiagnostic(
          category=StaticSafetyCategory.DYNAMIC_AUTOGRAD,
          message="Dynamic Autograd invocation in model execution path.",
          recommendation="Separate backward differentiation passes from pure static forward representations.",
          snippet=str(func_name),
        )
      )

  @classmethod
  def scan(cls, code_or_module: Union[str, cst.Module]) -> List[StaticSafetyDiagnostic]:
    """Scan source code or CST module for static safety boundaries.

    Args:
        code_or_module: Python source code string or parsed LibCST Module.

    Returns:
        List of detected StaticSafetyDiagnostic objects.
    """
    if isinstance(code_or_module, str):
      module = cst.parse_module(code_or_module)
    else:
      module = code_or_module

    scanner = cls()
    module.visit(scanner)
    return scanner.diagnostics
