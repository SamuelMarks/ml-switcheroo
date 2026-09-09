"""LibCST Visitors for structural parsing of ML frameworks.

This module provides visitors to extract relevant state and execution logic
from PyTorch, Keras, JAX, and MLX code using LibCST.
"""

from typing import Dict, List, Optional
import libcst as cst


class PyTorchVisitor(cst.CSTVisitor):
  """Visitor to extract state and logic from PyTorch models.

  Extracts:
      - state_allocations: Variables initialized with nn.Parameter or register_buffer.
      - forward_nodes: AST nodes within the forward method.
  """

  def __init__(self) -> None:
    """Initialize the PyTorchVisitor."""
    super().__init__()
    self.in_init: bool = False
    self.in_forward: bool = False
    self.state_allocations: Dict[str, cst.CSTNode] = {}
    self.forward_nodes: List[cst.CSTNode] = []

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Track whether we are in __init__ or forward methods.

    Args:
        node: The function definition node.

    Returns:
        None to continue traversal.
    """
    if node.name.value == "__init__":
      self.in_init = True
    elif node.name.value == "forward":
      self.in_forward = True
    return None

  def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
    """Exit the tracking of __init__ or forward methods.

    Args:
        original_node: The function definition node that was visited.
    """
    if original_node.name.value == "__init__":
      self.in_init = False
    elif original_node.name.value == "forward":
      self.in_forward = False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """Extract state allocations in __init__.

    Args:
        node: The assignment node.

    Returns:
        None to continue traversal.
    """
    if self.in_init and isinstance(node.value, cst.Call):
      call_node = node.value
      func = call_node.func

      is_parameter = False
      is_register_buffer = False

      if isinstance(func, cst.Attribute):
        if func.attr.value == "Parameter":
          is_parameter = True
      elif isinstance(func, cst.Name):
        if func.value == "Parameter":
          is_parameter = True

      if isinstance(func, cst.Attribute):
        if func.attr.value == "register_buffer":
          is_register_buffer = True
      elif isinstance(func, cst.Name):
        if func.value == "register_buffer":
          is_register_buffer = True

      if is_parameter:
        for target in node.targets:
          target_node = target.target
          if (
            isinstance(target_node, cst.Attribute)
            and isinstance(target_node.value, cst.Name)
            and target_node.value.value == "self"
          ):
            self.state_allocations[target_node.attr.value] = call_node
      elif is_register_buffer and len(call_node.args) >= 1:
        arg_name_node = call_node.args[0].value
        if isinstance(arg_name_node, cst.SimpleString):
          name = arg_name_node.value.strip("\"'")
          if len(call_node.args) >= 2:
            self.state_allocations[name] = call_node.args[1].value
    return None

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """Capture forward execution graph nodes (statements) or register_buffer calls.

    Args:
        node: The statement node.

    Returns:
        None to continue traversal.
    """
    if self.in_forward:
      self.forward_nodes.append(node)
    elif self.in_init:
      for stmt in node.body:
        if isinstance(stmt, cst.Expr) and isinstance(stmt.value, cst.Call):
          call_node = stmt.value
          func = call_node.func
          is_register_buffer = False
          if isinstance(func, cst.Attribute) and func.attr.value == "register_buffer":
            if isinstance(func.value, cst.Name) and func.value.value == "self":
              is_register_buffer = True
          if is_register_buffer and len(call_node.args) >= 1:
            arg_name_node = call_node.args[0].value
            if isinstance(arg_name_node, cst.SimpleString):
              # strip quotes
              name = arg_name_node.value.strip("\"'")
              if len(call_node.args) >= 2:
                self.state_allocations[name] = call_node.args[1].value
    return None


class KerasVisitor(cst.CSTVisitor):
  """Visitor to extract state and logic from Keras models.

  Extracts:
      - state_allocations: Variables initialized in build() or __init__.
      - call_nodes: AST nodes within the call() method.
  """

  def __init__(self) -> None:
    """Initialize the KerasVisitor."""
    super().__init__()
    self.in_init_or_build: bool = False
    self.in_call: bool = False
    self.state_allocations: Dict[str, cst.CSTNode] = {}
    self.call_nodes: List[cst.CSTNode] = []

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Track whether we are in __init__, build, or call methods.

    Args:
        node: The function definition node.

    Returns:
        None to continue traversal.
    """
    if node.name.value in ("__init__", "build"):
      self.in_init_or_build = True
    elif node.name.value == "call":
      self.in_call = True
    return None

  def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
    """Exit the tracking of methods.

    Args:
        original_node: The function definition node.
    """
    if original_node.name.value in ("__init__", "build"):
      self.in_init_or_build = False
    elif original_node.name.value == "call":
      self.in_call = False

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """Extract state allocations in __init__ or build.

    Args:
        node: The assignment node.

    Returns:
        None to continue traversal.
    """
    if self.in_init_or_build:
      for target in node.targets:
        target_node = target.target
        if (
          isinstance(target_node, cst.Attribute)
          and isinstance(target_node.value, cst.Name)
          and target_node.value.value == "self"
        ):
          self.state_allocations[target_node.attr.value] = node.value
    return None

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """Capture call execution graph nodes.

    Args:
        node: The statement node.

    Returns:
        None to continue traversal.
    """
    if self.in_call:
      self.call_nodes.append(node)
    return None


class JAXVisitor(cst.CSTVisitor):
  """Visitor to extract logic from JAX code.

  Extracts:
      - jit_nodes: Functions wrapped in jax.jit.
      - vmap_nodes: Functions wrapped in jax.vmap.
  """

  def __init__(self) -> None:
    """Initialize the JAXVisitor."""
    super().__init__()
    self.jit_nodes: List[cst.FunctionDef] = []
    self.vmap_nodes: List[cst.FunctionDef] = []

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Check for jit and vmap decorators.

    Args:
        node: The function definition node.

    Returns:
        None to continue traversal.
    """
    for decorator in node.decorators:
      dec_node = decorator.decorator
      if isinstance(dec_node, cst.Attribute) and dec_node.attr.value == "jit":
        self.jit_nodes.append(node)
      elif isinstance(dec_node, cst.Call):
        func = dec_node.func
        if isinstance(func, cst.Attribute) and func.attr.value == "jit":
          self.jit_nodes.append(node)
        elif isinstance(func, cst.Name) and func.value == "jit":
          self.jit_nodes.append(node)
        elif isinstance(func, cst.Attribute) and func.attr.value == "vmap":
          self.vmap_nodes.append(node)
        elif isinstance(func, cst.Name) and func.value == "vmap":
          self.vmap_nodes.append(node)
      elif isinstance(dec_node, cst.Name) and dec_node.value == "jit":
        self.jit_nodes.append(node)
      elif isinstance(dec_node, cst.Name) and dec_node.value == "vmap":
        self.vmap_nodes.append(node)
      elif isinstance(dec_node, cst.Attribute) and dec_node.attr.value == "vmap":
        self.vmap_nodes.append(node)
    return None


class MLXVisitor(cst.CSTVisitor):
  """Visitor to extract logic from MLX code."""

  def __init__(self) -> None:
    """Initialize the MLXVisitor."""
    super().__init__()
    self.state_allocations: Dict[str, cst.CSTNode] = {}

  def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
    """Extract generic state allocations for MLX.

    Args:
        node: The assignment node.

    Returns:
        None to continue traversal.
    """
    for target in node.targets:
      target_node = target.target
      if (
        isinstance(target_node, cst.Attribute)
        and isinstance(target_node.value, cst.Name)
        and target_node.value.value == "self"
      ):
        self.state_allocations[target_node.attr.value] = node.value
    return None


class ASTNormalizer(cst.CSTTransformer):
  """Normalizes variable names and scope across the AST."""

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
    """Standardize scope resolutions if needed.

    Args:
        original_node: The original name node.
        updated_node: The updated name node.

    Returns:
        The potentially modified name node.
    """
    return updated_node
