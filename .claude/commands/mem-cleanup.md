---
description: "Cleanup old memory system data"
tools:
  - "Bash"
---

# Cleanup Old Data

Clean up old memory system data based on retention policies.

```bash
python .prsist/prsist.py -z
```

This performs:
- Removal of expired sessions
- Archive of old data
- Database optimization
- Cleanup of temporary files
- Performance optimization

Note: This respects retention policies defined in system configuration.