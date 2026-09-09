"""State pivoting passes for translating between OOP and functional patterns.

These transformers handle moving state attributes (`self.weight`) to function
arguments (PyTrees) and vice versa, as well as PRNG key threading.
"""

from typing import Dict, Union
import libcst as cst


class OOPToFunctionalPivoter(cst.CSTTransformer):
  """Lifts OOP state into functional arguments.

  Transforms `self.weight` to `params['weight']` or simply `weight` if unpacked.
  Injects `jax.random.PRNGKey` arguments into stochastic function signatures.
  """

  def __init__(self, state_mapping: Dict[str, str], inject_rng: bool = False) -> None:
    """Initialize the pivoter.

    Args:
        state_mapping: Mapping of attribute names (e.g., 'weight') to PyTree paths (e.g., 'params["weight"]').
                       For simplicity, this implementation maps to top-level variables.
        inject_rng: If True, injects an `rng` argument into function definitions.
    """
    super().__init__()
    self.state_mapping = state_mapping
    self.inject_rng = inject_rng

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> Union[cst.Attribute, cst.Name]:
    """Convert `self.var_name` to `mapped_name`.

    Args:
        original_node: The original attribute node.
        updated_node: The updated attribute node.

    Returns:
        A new Name node if mapped, else the updated Attribute.
    """
    if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
      attr_name = updated_node.attr.value
      if attr_name in self.state_mapping:
        return cst.Name(value=self.state_mapping[attr_name])
    return updated_node

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    """Inject RNG argument into function signature if required.

    Args:
        original_node: Original function node.
        updated_node: Updated function node.

    Returns:
        Function with injected argument if inject_rng is True.
    """
    if self.inject_rng and updated_node.name.value in ("forward", "call", "__call__"):
      # Check if `rng` is already there
      has_rng = any(param.name.value == "rng" for param in updated_node.params.params)
      if not has_rng:
        new_param = cst.Param(name=cst.Name("rng"))
        new_params = list(updated_node.params.params)
        new_params.append(new_param)
        return updated_node.with_changes(params=updated_node.params.with_changes(params=new_params))
    return updated_node


class FunctionalToOOPPivoter(cst.CSTTransformer):
  """Maps functional PyTree parameters back into OOP state attributes.

  Transforms `weight` back to `self.weight`.
  Can drop explicitly threaded `rng` variables.
  """

  def __init__(self, state_mapping: Dict[str, str], drop_rng: bool = False) -> None:
    """Initialize the pivoter.

    Args:
        state_mapping: Mapping of functional variable names to `self` attributes.
        drop_rng: If True, drops `rng` argument and uses global state equivalent.
    """
    super().__init__()
    self.state_mapping = state_mapping
    self.drop_rng = drop_rng

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> Union[cst.Name, cst.Attribute]:
    """Convert mapped local variables to `self.attr_name`.

    Args:
        original_node: Original name node.
        updated_node: Updated name node.

    Returns:
        An Attribute node if mapped, else the updated Name.
    """
    if updated_node.value in self.state_mapping:
      return cst.Attribute(value=cst.Name(value="self"), attr=cst.Name(value=self.state_mapping[updated_node.value]))
    return updated_node

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    """Drop RNG argument from function signature.

    Args:
        original_node: Original function node.
        updated_node: Updated function node.

    Returns:
        Function with RNG argument dropped if drop_rng is True.
    """
    if self.drop_rng:
      new_params = []
      for i, p in enumerate(updated_node.params.params):
        if p.name.value != "rng":
          # Keep comma unless it's the last element we are keeping
          new_params.append(p)

      # Properly clean up commas
      if new_params:
        new_params[-1] = new_params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

      if len(new_params) != len(updated_node.params.params):
        return updated_node.with_changes(params=updated_node.params.with_changes(params=new_params))
    return updated_node
