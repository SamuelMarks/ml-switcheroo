#!/usr/bin/env bash
# ==============================================================================
# ml-switcheroo: Knowledge Base Synchronization Script
# ==============================================================================
#
# This script iterates over all registered Framework Adapters in the
# ml-switcheroo package and runs the `sync` command for each.
#
# It updates the Snapshot Overlays (snapshots/*_map.json) by finding concrete
# implementations for the Abstract Standards defined in the Hub.
#
# Usage:
#   ./maintenance/sync_all.sh
# ==============================================================================

set -o pipefail

# Visual helpers
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}🔄 ml-switcheroo Knowledge Base Sync${NC}"
echo "----------------------------------------"

# 1. Detect Python Environment
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

if ! $PYTHON_CMD -c "import ml_switcheroo" &> /dev/null; then
    echo -e "${RED}❌ Error: 'ml_switcheroo' package not found in python environment.${NC}"
    echo "   Please run: pip install -e ."
    exit 1
fi

# 2. Dynamically Discover Support Frameworks
# We ask the package what adapters are registered.
echo -e "🔍 Discovering registered frameworks..."
FRAMEWORKS=$($PYTHON_CMD -c "from ml_switcheroo.frameworks import available_frameworks; print(' '.join(available_frameworks()))")

if [ -z "$FRAMEWORKS" ]; then
    echo -e "${RED}❌ Error: No frameworks returned from registry.${NC}"
    exit 1
fi

echo -e "   Found: ${YELLOW}${FRAMEWORKS}${NC}"
echo ""

# 3. Iterate and Sync
FAILED=0
SUCCESS=0

for fw in $FRAMEWORKS; do
    echo -e "${BOLD}👉 Syncing ${fw}...${NC}"

    # We allow the sync command to fail/skip gracefully without stopping this script
    # The sync command returns 0 even if it finds nothing, unless it crashes.
    if ml_switcheroo sync "$fw"; then
        echo -e "   ${GREEN}✔ Done${NC}"
        ((SUCCESS++))
    else
        echo -e "   ${RED}✘ Failed${NC}"
        ((FAILED++))
    fi
    echo ""
done

# 4. Summary
echo "----------------------------------------"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ All $SUCCESS frameworks synced successfully.${NC}"
    echo "   Verify changes in src/ml_switcheroo/snapshots/"
    exit 0
else
    echo -e "${YELLOW}⚠️  Completed with issues: $SUCCESS passed, $FAILED failed.${NC}"
    exit 1
fi
