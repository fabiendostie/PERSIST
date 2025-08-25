---
description: "Show cross-session correlations and connections"
tools:
  - "Bash"
---

# Cross-Session Correlation

Analyze connections and patterns across different sessions.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from cross_session_correlator import CrossSessionCorrelator
correlator = CrossSessionCorrelator('.prsist', '.')
print('Cross-Session Correlation Analysis:')
correlations = correlator.find_session_correlations()
print(f'Found {len(correlations)} correlations:')
for corr in correlations[:5]:  # Show first 5
    print(f'  - Sessions {corr.get(\"session1\", \"?\")[:8]} <-> {corr.get(\"session2\", \"?\")[:8]}')
    print(f'    Similarity: {corr.get(\"similarity_score\", 0):.2f}')
    print(f'    Common: {corr.get(\"common_elements\", [])}')
"
```

Analyzes:
- Related sessions and work patterns
- Similar problem-solving approaches
- Code evolution across sessions
- Recurring development themes
- Session clustering and grouping