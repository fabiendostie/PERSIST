---
description: "Start a new memory system session"
tools:
  - "Bash"
---

# Start New Session

Start a new memory system session manually (normally handled automatically by SessionStart hook).

```bash
python .prsist/prsist.py -n
```

This creates:
- New session with unique ID
- Initial session state
- Context capture point
- Git correlation data

Note: Sessions normally start automatically when Claude Code launches.