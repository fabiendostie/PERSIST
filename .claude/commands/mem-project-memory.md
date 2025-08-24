---
description: "Add information to project memory"
tools:
  - "Bash"
---

# Add to Project Memory

Add important information directly to project memory for long-term persistence.

```bash
python claude-commands.py decision "$ARGUMENTS"
```

Usage: `/mem-project-memory "Information to add to project memory"`

Examples:
- `/mem-project-memory "API endpoint changed to /api/v2/"`
- `/mem-project-memory "Database migration completed successfully"`
- `/mem-project-memory "New coding standards adopted for TypeScript"`

This adds information to the persistent project memory file that's loaded in every session.