---
description: "Access cross-session knowledge and learning patterns"
tools:
  - "Bash"
---

# Knowledge Manager

Access accumulated knowledge and learning patterns across sessions.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from knowledge_manager import KnowledgeManager
km = KnowledgeManager('.prsist/storage')
print('Knowledge Base Summary:')
patterns = km.get_learning_patterns()
insights = km.get_accumulated_insights()
print('Learning Patterns:', len(patterns))
for pattern in patterns[:5]:  # Show first 5
    print(f'  - {pattern.get(\"pattern_type\", \"Unknown\")}: {pattern.get(\"description\", \"N/A\")}')
print('Accumulated Insights:', len(insights))
for insight in insights[:5]:  # Show first 5
    print(f'  - {insight.get(\"category\", \"General\")}: {insight.get(\"content\", \"N/A\")[:80]}...')
"
```

Provides:
- Cross-session learning patterns
- Accumulated project knowledge
- Decision history and reasoning
- Pattern-based recommendations
- Knowledge persistence insights