#!/bin/sh
# ==============================================================================
# ml-switcheroo: Knowledge Base Bootstrap
# ==============================================================================
#
# "Zero-to-Hero" regeneration script.
# 1. DELETE all existing semantic definitions and mappings.
# 2. FETCH upstream standards (Array API, ONNX).
# 3. IMPORT standards into the Knowledge Base (Hub).
# 4. SYNC installed frameworks to generate new mappings (Spokes).
#
# Usage:
#   ./scripts/bootstrap.sh
# ==============================================================================

set -e  # Exit on error

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ROOT_DIR="$(dirname -- "$(dirname -- "$(realpath -- "$0")")")"
VENV_DIR="${VENV_DIR:-$ROOT_DIR}"
TEMP_DIR="_bootstrap_temp"
SEMANTICS_DIR="$ROOT_DIR"'/src/ml_switcheroo/semantics'
SNAPSHOTS_DIR="$ROOT_DIR"'/src/ml_switcheroo/snapshots'

echo "${BOLD}🔥 Initiating Knowledge Base Bootstrap${NC}"
echo "----------------------------------------"

# ------------------------------------------------------------------------------
# 0. DESTRUCTIVE CLEANUP
# ------------------------------------------------------------------------------
echo "${BLUE}[0/4] Cleaning existing JSON artifacts...${NC}"
# Remove Specs (Hub)
rm -f "$SEMANTICS_DIR"/k_*.json
# Remove Mappings (Spokes)
rm -f "$SNAPSHOTS_DIR"/*_map.json

# Note: We purposely leave framework capture snapshots (e.g. torch_v2.0.json)
# to allow Ghost Mode to function if libraries aren't installed.
echo "   ${GREEN}✔ Cleaned $SEMANTICS_DIR${NC}"
echo "   ${GREEN}✔ Cleaned $SNAPSHOTS_DIR${NC}"

# 0b. Prepare Workspace
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# ------------------------------------------------------------------------------
# 1. IMPORT TIER A: MATH (Array API Standard)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[1/4] Importing Tier A (Array API)...${NC}"
echo "   Fetching upstream data-apis/array-api..."

# Clone specific compatible version to ensure stability
git clone --depth 1 --branch 2024.12 https://github.com/data-apis/array-api "$TEMP_DIR/array-api" > /dev/null 2>&1

echo "   Parsing stubs..."
# Path relative to repo root: src/array_api_stubs/_2024_12
ml_switcheroo import-spec "$TEMP_DIR/array-api/src/array_api_stubs/_2024_12"

# ------------------------------------------------------------------------------
# 2. IMPORT TIER B: NEURAL (ONNX Operators)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[2/4] Importing Tier B (ONNX Neural)...${NC}"
echo "   Fetching upstream onnx/onnx..."

# Clone specific compatible version
git clone --depth 1 --branch v1.20.0 https://github.com/onnx/onnx "$TEMP_DIR/onnx" > /dev/null 2>&1

echo "   Parsing Operators.md..."
ml_switcheroo import-spec "$TEMP_DIR/onnx/docs/Operators.md"

# ------------------------------------------------------------------------------
# 3. IMPORT TIER C: INTERNAL (Extras & Transforms)
# ------------------------------------------------------------------------------
echo "\n${BLUE}[3/4] Importing Tier C (Internal Standards)...${NC}"
# Uses the 'internal' keyword mapped in handlers/discovery.py
ml_switcheroo import-spec internal

# ------------------------------------------------------------------------------
# 4. IMPLEMENTATION SYNC
# ------------------------------------------------------------------------------
echo "\n${BLUE}[4/4] Syncing Installed Frameworks...${NC}"

# Iterate standard supported frameworks
# We allow failure (|| true) so the script finishes even if a lib isn't installed.
if [ -d "$VENV_DIR"'/.venv-pyenv-3-12' ] || [ -d "$VENV_DIR"'/.venv-uv-3-12' ]; then
  if [ -d "$VENV_DIR"'/.venv-pyenv-3-12' ]; then
        . "$VENV_DIR"'/.venv-pyenv-3-12/bin/activate'
        python3 -m pip install -e .  > /dev/null 2>&1
        python3 -m pip install torch jax numpy tensorflow keras mlx > /dev/null 2>&1
    else
        . "$VENV_DIR"'/.venv-uv-3-12/bin/activate'
        uv pip install -e . > /dev/null 2>&1
        uv pip install torch jax numpy tensorflow keras mlx > /dev/null 2>&1
    fi
  for fw in torch jax numpy tensorflow keras mlx; do
      echo -n "   👉 Syncing $fw... "
      if ml_switcheroo sync "$fw" > /dev/null 2>&1; then
          echo "${GREEN}✔ Done${NC}"
      else
          echo "${RED}Failed${NC}"
      fi
  done
fi
if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ] || [ -d "$VENV_DIR"'/.venv-uv-3-10' ]; then
    echo -n "   👉 Syncing paxml... "
    if [ -d "$VENV_DIR"'/.venv-pyenv-3-10' ]; then
        . "$VENV_DIR"'/.venv-pyenv-3-10/bin/activate'
        python3 -m pip install -e .  > /dev/null 2>&1
        python3 -m pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1
    else
        . "$VENV_DIR"'/.venv-uv-3-10/bin/activate'
        uv pip install -e . > /dev/null 2>&1
        uv pip install paxml jaxlib=='0.4.26' > /dev/null 2>&1
    fi
    if ml_switcheroo sync paxml > /dev/null 2>&1; then
        echo "${GREEN}✔ Done${NC}"
    else
        echo "${RED}Failed${NC}"
    fi
else
    echo "${YELLOW}Skipping paxml"
fi

# ------------------------------------------------------------------------------
# CLEANUP & SUMMARY
# ------------------------------------------------------------------------------
echo "\n${BOLD}🧹 Cleanup...${NC}"
rm -rf "$TEMP_DIR"

echo "----------------------------------------"
echo "${GREEN}✅ Bootstrap Complete.${NC}"
echo "   Verify generated specs in: $SEMANTICS_DIR"
echo "   Verify generated maps in:  $SNAPSHOTS_DIR"
