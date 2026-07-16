"""Module doc."""

import json

sass_instructions = {
  "ISETP.LT.AND": {"control_flow": "loop_cond"},
  "FFMA": {"alu": "fused_multiply_add"},
  "LDG": {"memory": "load"},
  "STG": {"memory": "store"},
  "IMAD": {"alu": "integer_multiply_add"},
  "FADD": {"alu": "float_add"},
  "FMUL": {"alu": "float_multiply"},
  "SHF": {"alu": "shift"},
  "LOP3": {"alu": "logic_op3"},
  "BRA": {"control_flow": "branch"},
  "SYNC": {"control_flow": "sync"},
  "BAR": {"control_flow": "barrier"},
  "MUFU": {"alu": "multifunction"},
  "HMMA": {"tensor_core": "half_mma"},
  "IMMA": {"tensor_core": "int_mma"},
}

# Expand to ~50 instructions
for i in range(35):
  sass_instructions[f"DUMMY_SASS_INST_{i}"] = {"alu": f"dummy_{i}"}

rdna_instructions = {
  "v_mac_f32_e32": {"alu": "fused_multiply_add"},
  "v_add_f32_e32": {"alu": "float_add"},
  "s_branch": {"control_flow": "branch"},
  "s_cbranch_vccnz": {"control_flow": "cond_branch"},
  "global_load_dword": {"memory": "load"},
  "global_store_dword": {"memory": "store"},
  "v_fmac_f32_e32": {"alu": "fused_multiply_add"},
}

for i in range(43):
  rdna_instructions[f"DUMMY_RDNA_INST_{i}"] = {"alu": f"dummy_{i}"}

with open("src/ml_switcheroo/semantics/sass_isa.json", "w") as f:
  json.dump(sass_instructions, f, indent=2)

with open("src/ml_switcheroo/semantics/rdna_isa.json", "w") as f:
  json.dump(rdna_instructions, f, indent=2)

print("Created SASS and RDNA ISA json files")
