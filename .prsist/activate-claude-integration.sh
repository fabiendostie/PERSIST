#!/bin/bash
# Activate Prsist Memory System for Claude Code
# This script can be called automatically when Claude Code starts

echo "Activating Prsist Memory System for Claude Code..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the integration script
python3 "$SCRIPT_DIR/bin/claude-integration.py"

# Set environment variables to indicate prsist is active
export PRSIST_ACTIVE=true
export PRSIST_CONTEXT_FILE="$SCRIPT_DIR/context/claude-context.md"

# Optional: Display context file location for reference
if [ -f "$PRSIST_CONTEXT_FILE" ]; then
    echo "📄 Project context available at: $PRSIST_CONTEXT_FILE"
else
    echo "🔄 Context file will be created when needed"
fi

echo ""
echo "🧠 Prsist Memory System is now active for your Claude Code session"
echo ""