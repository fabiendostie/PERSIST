!---
description: "Run all core memory system checks (test, status, context, health, recent, memory, validate)"
tools:
  - "Bash"
---

# Run All Core Checks

Run all core memory system checks in one command: test, status, context, health, recent, memory, and validate.

```bash
python .prsist/prsist.py -a
```

This executes all core operations:
- System test (-t)
- Session status (-s)
- Context display (-c)
- Health check (-h)
- Recent sessions (-r)
- Memory statistics (-m)
- System validation (-v)

Perfect for comprehensive system verification.