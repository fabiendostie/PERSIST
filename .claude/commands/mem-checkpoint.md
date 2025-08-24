---
description: "Create a manual checkpoint in current session"
tools:
  - "Bash"
---

# Create Checkpoint

Create a manual checkpoint to mark progress in the current development session.

```bash
python claude-commands.py feature "Checkpoint" "Manual checkpoint - $ARGUMENTS"
```

Usage: `/mem-checkpoint "Description of current progress"`

Examples:
- `/mem-checkpoint "Completed database migration"`
- `/mem-checkpoint "Finished API integration testing"`
- `/mem-checkpoint "Ready for code review"`

This creates a checkpoint with your description and preserves the current session state.