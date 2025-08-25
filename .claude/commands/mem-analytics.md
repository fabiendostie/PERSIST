---
description: "Show advanced analytics and insights from memory data"
tools:
  - "Bash"
---

# Advanced Analytics

Generate insights from memory data using Phase 4 analytics engine.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from optimization.analytics_engine import AnalyticsEngine
engine = AnalyticsEngine('.prsist/storage/sessions.db')
print('Generating analytics insights...')
insights = engine.generate_comprehensive_insights()
print('Analytics Insights:')
for category, data in insights.items():
    print(f'{category}:')
    if isinstance(data, dict):
        for k, v in data.items():
            print(f'  {k}: {v}')
    else:
        print(f'  {data}')
"
```

Shows:
- Project health patterns
- Development bottleneck analysis  
- Team collaboration insights
- Memory system optimization suggestions
- Predictive development trends