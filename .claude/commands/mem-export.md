---
description: "Export current session data"
tools:
  - "Bash"
---

# Export Session Data

Export current session data for backup or analysis.

```bash
python .prsist/prsist.py -x
```

This exports:
- Session metadata and timeline
- Tool usage history
- File modification records
- Context snapshots
- Performance metrics

Data is exported in JSON format for easy processing.