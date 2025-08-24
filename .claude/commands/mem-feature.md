---
description: "Log a completed feature with automatic checkpoint"
tools:
  - "Bash"
---

# Log Feature Completion

Log a completed feature to the memory system with automatic checkpoint creation.

```bash
python claude-commands.py feature "$ARGUMENTS"
```

Usage: `/mem-feature "Feature Name" "Description of what was completed"`

Examples:
- `/mem-feature "User Authentication" "Implemented JWT-based login system"`
- `/mem-feature "API Integration" "Connected to payment service"`
- `/mem-feature "Bug Fix #123" "Fixed memory leak in processing"`

This automatically:
- Logs the feature to project memory
- Creates a checkpoint for the session
- Updates the development history