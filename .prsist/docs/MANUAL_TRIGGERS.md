# Manual Triggers and Commands Reference

This document provides a comprehensive guide to all manual triggers, commands, and operations available in the Prsist Memory System.

## 🎯 Quick Command Reference

### Essential Commands

```bash
# Test the complete memory system
python .prsist/test_system.py

# Get memory system status and statistics
python .prsist/memory_manager.py status

# Log a feature completion (creates checkpoint)
python .prsist/hooks/FeatureComplete.py "Feature Name" "Description"

# Start a manual session
python .prsist/session_tracker.py start

# End current session and archive it
python .prsist/session_tracker.py end

# Create a manual checkpoint
python .prsist/session_tracker.py checkpoint "checkpoint_name"
```

## 🔧 Hook Triggers

### Automatic Hooks (Configured in Claude Code)

```bash
# SessionStart - Triggers when Claude Code starts
# Configured in .claude/settings.local.json
python .prsist/hooks/SessionStart.py
```

### Manual Hooks

```bash
# Feature completion logging
python .prsist/hooks/FeatureComplete.py "Feature Name" "Optional description"

# Example usage:
python .prsist/hooks/FeatureComplete.py "User Authentication" "Implemented JWT-based auth system with refresh tokens"
python .prsist/hooks/FeatureComplete.py "Database Migration" "Added user preferences table and indexes"
python .prsist/hooks/FeatureComplete.py "Bug Fix - Memory Leak" "Fixed session cleanup in memory manager"
```

## 📊 Memory Management Commands

### Session Operations

```bash
# Start new session with context
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
result = mm.start_session({'context': 'Manual session start', 'task': 'Development work'})
print(f'Session ID: {result[\"session_id\"]}')
"

# Get current session info
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
info = mm.get_session_info()
print(f'Session: {info.get(\"session_id\", \"None\")}')
print(f'Tools used: {info.get(\"tool_usage_count\", 0)}')
print(f'Files modified: {info.get(\"file_interaction_count\", 0)}')
"

# End current session
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
result = mm.end_session(archive=True)
print(f'Session ended: {result}')
"
```

### Context and Memory Operations

```bash
# Get current session context (what Claude sees)
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
context = mm.get_session_context()
print(context)
"

# Add to project memory
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
mm.add_project_memory('New important project information or decision')
print('Project memory updated')
"

# Add a decision record
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
mm.add_decision(
    title='Architecture Decision',
    description='Decided to use SQLite for local storage instead of JSON files',
    category='architecture',
    impact='high'
)
print('Decision recorded')
"
```

### Database Operations

```bash
# Get recent sessions
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
sessions = mm.get_recent_sessions(limit=5)
for session in sessions:
    print(f'{session.get(\"start_time\", \"Unknown\")} - {session.get(\"session_id\", \"Unknown\")}')
"

# Get memory system statistics
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
stats = mm.get_memory_stats()
print(f'Total sessions: {stats.get(\"total_sessions\", 0)}')
print(f'Database size: {stats.get(\"database_size_mb\", 0)} MB')
print(f'Active session: {stats.get(\"active_session\", \"None\")}')
"
```

## 🧹 Maintenance Commands

### Cleanup Operations

```bash
# Clean up old data (default: 30 days)
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
cleanup_stats = mm.cleanup_old_data(retention_days=30)
print(f'Cleaned up: {cleanup_stats}')
"

# Validate system integrity
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
validation = mm.validate_system()
print(f'System valid: {validation[\"valid\"]}')
if not validation['valid']:
    print(f'Issues: {validation[\"issues\"]}')
"
```

### Export and Backup

```bash
# Export current session data
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
data = mm.export_session_data(format='json')
if data:
    with open('session_export.json', 'w') as f:
        f.write(data)
    print('Session exported to session_export.json')
else:
    print('No active session to export')
"

# Export specific session
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
# Replace with actual session ID
session_id = 'your-session-id-here'
data = mm.export_session_data(session_id=session_id, format='json')
if data:
    with open(f'session_{session_id[:8]}_export.json', 'w') as f:
        f.write(data)
    print(f'Session {session_id[:8]} exported')
"
```

## 🔍 Debugging and Inspection

### System Diagnostics

```bash
# Check hook file existence and permissions
ls -la .prsist/hooks/
ls -la .claude/settings.local.json

# Test hook execution directly
python .prsist/hooks/SessionStart.py
python .prsist/hooks/FeatureComplete.py "Test Feature" "Testing hook execution"

# Check database connection
python -c "
from database import MemoryDatabase
db = MemoryDatabase()
sessions = db.get_recent_sessions(1)
print(f'Database connected, found {len(sessions)} sessions')
"

# Inspect configuration
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
import json
print(json.dumps(mm.config, indent=2))
"
```

### Log Analysis

```bash
# View recent log entries (if logging to file)
tail -f .prsist/logs/memory_system.log

# Check for errors in recent logs
grep ERROR .prsist/logs/memory_system.log | tail -10
```

## 📋 Integration Commands

### Git Integration

```bash
# Get current git information
python -c "
from utils import get_git_info
git_info = get_git_info()
print(f'Branch: {git_info.get(\"branch\", \"unknown\")}')
print(f'Hash: {git_info.get(\"hash\", \"unknown\")}')
print(f'Status: {git_info.get(\"status\", \"unknown\")}')
"
```

### Project Context

```bash
# Get project root and basic info
python -c "
from utils import get_project_root
root = get_project_root()
print(f'Project root: {root}')
"

# Update project memory with current state
python .prsist/hooks/FeatureComplete.py "Project Checkpoint" "$(date): Manual checkpoint creation"
```

## 🎛️ Configuration Commands

### Memory System Configuration

```bash
# View current configuration
cat .prsist/config/memory-config.yaml

# Validate configuration
python -c "
from utils import load_yaml_config
config = load_yaml_config('.prsist/config/memory-config.yaml')
print('Configuration loaded successfully')
print(f'Sections: {list(config.keys())}')
"
```

### Claude Code Configuration

```bash
# View current hook configuration
cat .claude/settings.local.json | grep -A 10 hooks

# Test Claude Code settings validation
# (This would be done by Claude Code itself when it starts)
```

## 🚀 Workflow Examples

### Feature Development Workflow

```bash
# 1. Start development (automatic via Claude Code SessionStart hook)

# 2. During development - log major milestones
python .prsist/hooks/FeatureComplete.py "API Endpoint Created" "POST /api/users endpoint with validation"

# 3. After testing
python .prsist/hooks/FeatureComplete.py "Tests Added" "Unit tests for user creation with 95% coverage"

# 4. After deployment
python .prsist/hooks/FeatureComplete.py "Feature Deployed" "User management feature live in production"

# 5. Session ends automatically when Claude Code closes
```

### Debugging Workflow

```bash
# 1. Check system status
python .prsist/test_system.py

# 2. Review recent sessions for context
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
sessions = mm.get_recent_sessions(3)
for s in sessions:
    print(f'{s.get(\"start_time\")}: {s.get(\"summary\", \"No summary\")}')
"

# 3. Export problem session for analysis
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
data = mm.export_session_data()
with open('debug_session.json', 'w') as f:
    f.write(data)
"
```

### Maintenance Workflow

```bash
# Weekly maintenance
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()

# Validate system
validation = mm.validate_system()
print(f'System health: {\"OK\" if validation[\"valid\"] else \"ISSUES\"}')

# Clean up old data
cleanup = mm.cleanup_old_data(retention_days=30)
print(f'Cleanup: {cleanup}')

# Get stats
stats = mm.get_memory_stats()
print(f'Total sessions: {stats.get(\"total_sessions\", 0)}')
print(f'Database size: {stats.get(\"database_size_mb\", 0)} MB')
"
```

## 📱 Quick Reference Card

| Operation | Command |
|-----------|---------|
| Test System | `python .prsist/test_system.py` |
| Log Feature | `python .prsist/hooks/FeatureComplete.py "Name" "Description"` |
| Get Context | `python -c "from memory_manager import MemoryManager; print(MemoryManager().get_session_context())"` |
| Session Info | `python -c "from memory_manager import MemoryManager; print(MemoryManager().get_session_info())"` |
| Recent Sessions | `python -c "from memory_manager import MemoryManager; print(MemoryManager().get_recent_sessions(5))"` |
| System Stats | `python -c "from memory_manager import MemoryManager; print(MemoryManager().get_memory_stats())"` |
| Validate System | `python -c "from memory_manager import MemoryManager; print(MemoryManager().validate_system())"` |
| Cleanup Data | `python -c "from memory_manager import MemoryManager; print(MemoryManager().cleanup_old_data())"` |

## 🔧 Troubleshooting Commands

### Common Issues

```bash
# Hook not executing
ls -la .claude/settings.local.json
python .prsist/hooks/SessionStart.py  # Test directly

# Database issues
python -c "from database import MemoryDatabase; MemoryDatabase().get_recent_sessions(1)"

# Permission issues
chmod +x .prsist/hooks/*.py

# Configuration issues
python -c "from memory_manager import MemoryManager; MemoryManager().validate_system()"
```

This reference covers all manual triggers and commands available in the memory system. Save this document for quick access to any memory system operation.