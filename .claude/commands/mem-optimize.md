---
description: "Run KV-cache optimization and performance tuning"
tools:
  - "Bash"
---

# KV-Cache Optimization

Run context caching optimization and performance analysis.

```bash
python -c "
import sys; sys.path.insert(0, '.prsist')
from optimization.kv_cache_manager import KVCacheManager
cache = KVCacheManager('.prsist/cache/kv_cache')
print('KV-Cache Optimization Status:')
stats = cache.get_cache_statistics()
print('Cache Statistics:')
for key, value in stats.items():
    print(f'  {key}: {value}')
print('\\nRunning optimization...')
cache.optimize_cache()
print('Optimization complete!')
"
```

Features:
- Context prefix caching
- Memory usage optimization
- Performance acceleration
- Cache hit rate analysis
- Automatic optimization suggestions