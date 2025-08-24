---
description: "Record an important decision or insight"
tools:
  - "Bash"
---

# Record Decision

Add an important decision or insight to the project memory for future reference.

```bash
python claude-commands.py decision "$ARGUMENTS"
```

Usage: `/mem-decision "Your decision or insight"`

Examples:
- `/mem-decision "Use Redis for caching instead of in-memory storage"`
- `/mem-decision "Switch to microservices for better scalability"`
- `/mem-decision "Implement rate limiting to prevent API abuse"`

This adds the decision to project memory so future AI sessions can reference it.