---
description: "Run semantic analysis on project code and sessions"
tools:
  - "Bash"
---

# Semantic Analysis

Analyze code semantics, patterns, and relationships across sessions.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from semantic_analyzer import SemanticAnalyzer
analyzer = SemanticAnalyzer('.prsist', '.')
print('Running semantic analysis...')
patterns = analyzer.analyze_project_semantics()
print('Semantic Analysis Results:')
for category, data in patterns.items():
    print(f'{category}: {len(data) if isinstance(data, list) else data}')
    if isinstance(data, list) and data:
        for item in data[:3]:  # Show first 3 items
            print(f'  - {item}')
"
```

Provides:
- Code semantic similarity analysis
- Cross-session pattern detection
- Function and class relationship mapping
- Code complexity metrics
- Semantic clustering insights