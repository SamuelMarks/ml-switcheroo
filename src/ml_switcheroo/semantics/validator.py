"""Validation models."""

import typing
from typing import Optional


class SemanticsValidator:
  """Schema validation for Operation definitions."""

  @staticmethod
  def validate_and_dump(abstract_id: str, new_data: dict[str, typing.Any]) -> Optional[dict[str, typing.Any]]:
    """Validate against the OperationDef model and return dumped data or None on failure.

    Args:
        abstract_id: Canonical abstract operation identifier.
        new_data: Raw operation definition dictionary to validate.

    Returns:
        Validated and serialized dictionary if valid, else None.
    """
    from pydantic import ValidationError

    from ml_switcheroo.core.dsl import OperationDef

    details_to_validate = new_data.copy()

    # 1. Inject missing fields required by Schema if not present
    if "operation" not in details_to_validate:
      details_to_validate["operation"] = abstract_id
    if "variants" not in details_to_validate:
      details_to_validate["variants"] = {}
    if "description" not in details_to_validate:
      details_to_validate["description"] = f"Definition for {abstract_id}"
    if "std_args" not in details_to_validate:
      details_to_validate["std_args"] = []

    try:
      validated = OperationDef.model_validate(details_to_validate)
      return validated.model_dump(by_alias=True, exclude_unset=True)
    except ValidationError as e:
      print(f"❌ Cannot update invalid definition for '{abstract_id}': {e}")
      return None
