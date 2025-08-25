---
description: "Show development productivity metrics and patterns"
tools:
  - "Bash"
---

# Productivity Tracker

Analyze development velocity, patterns, and memory system effectiveness.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from productivity_tracker import ProductivityTracker
tracker = ProductivityTracker()
metrics = tracker.get_productivity_report()
print('Development Productivity Report:')
for key, value in metrics.items():
    if isinstance(value, dict):
        print(f'{key}:')
        for k, v in value.items():
            print(f'  {k}: {v}')
    else:
        print(f'{key}: {value}')
"
```

Shows:
- Development velocity trends
- Session effectiveness patterns
- Tool usage analytics
- Memory system impact on productivity
- Collaboration patterns and recommendations