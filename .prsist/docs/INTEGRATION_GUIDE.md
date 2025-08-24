# Prsist Memory System - Complete Integration Guide

This document provides comprehensive documentation on how the memory system integrates with Claude Code and how to use it effectively.

## 🎯 Overview

The Prsist Memory System provides persistent memory across Claude Code sessions through:
- **Automatic session tracking** when Claude Code starts/stops
- **Manual feature completion logging** for major milestones
- **Context injection** with project history and decisions
- **Intelligent memory management** with cleanup and archiving

## 🔄 How Integration Works

### Automatic Integration

1. **Session Start Hook** (`.claude/settings.local.json`)
   - Triggers when Claude Code starts
   - Loads project context and recent session history
   - Injects memory into Claude's context automatically

2. **Background Memory Tracking**
   - Sessions tracked in SQLite database
   - Project memory maintained in markdown files
   - Context built from recent sessions and decisions

### Manual Integration Points

1. **Feature Completion Logging**
   - Log major milestones and completed features
   - Creates checkpoints for important project states
   - Updates project memory with achievements

2. **Decision Recording**
   - Document architectural and technical decisions  
   - Maintain project patterns and conventions
   - Build institutional knowledge over time

## 🚀 Getting Started

### Prerequisites

- Claude Code installed and configured
- Python 3.7+ available in PATH
- Project directory with `.claude/` folder

### Initial Setup

1. **Verify Installation**
   ```bash
   python .prsist/test_system.py
   ```

2. **Check Hook Configuration**
   ```bash
   cat .claude/settings.local.json
   ```
   Should contain:
   ```json
   {
     "hooks": {
       "SessionStart": [
         {
           "hooks": [
             {
               "type": "command",
               "command": "python .prsist/hooks/SessionStart.py"
             }
           ]
         }
       ]
     }
   }
   ```

3. **Test Integration**
   ```bash
   python .prsist/hooks/SessionStart.py
   ```

## 📊 Usage Patterns

### Daily Development Workflow

1. **Start Claude Code** (automatic)
   - Memory system initializes
   - Context loaded from previous sessions
   - Recent decisions and patterns available

2. **During Development**
   - Work normally with Claude Code
   - Memory system runs in background

3. **Log Major Features** (manual)
   ```bash
   python .prsist/hooks/FeatureComplete.py "User Login System" "Implemented OAuth2 with JWT tokens"
   ```

4. **End Session** (automatic)
   - Session archived when Claude Code closes
   - Context preserved for next session

### Feature Development Lifecycle

```bash
# Planning phase
python .prsist/hooks/FeatureComplete.py "Feature Planning" "User story analysis and technical design completed"

# Implementation phase  
python .prsist/hooks/FeatureComplete.py "Core Implementation" "Base functionality implemented and tested"

# Testing phase
python .prsist/hooks/FeatureComplete.py "Testing Complete" "Unit tests, integration tests, and QA validation passed"

# Deployment phase
python .prsist/hooks/FeatureComplete.py "Feature Deployed" "Feature successfully deployed to production"
```

### Architecture Decision Recording

```bash
# Using the Python API for more detailed decisions
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
mm.add_decision(
    title='Database Choice for User Sessions',
    description='Chose Redis over PostgreSQL for session storage due to TTL support and performance requirements',
    category='architecture',
    impact='high'
)
print('Architecture decision recorded')
"
```

## 🔧 Configuration Options

### Memory System Configuration

Location: `.prsist/config/memory-config.yaml`

Key settings:
```yaml
memory_system:
  auto_inject: true          # Automatically inject context
  max_context_length: 8000   # Maximum context size
  retention_days: 30         # Data retention period

session:
  auto_checkpoint_interval: 10  # Auto-checkpoint every N tools
  max_session_duration: 8      # Hours before session timeout

storage:
  database_path: "storage/sessions.db"
  backup_enabled: true
  compression_enabled: false
```

### Claude Code Hook Configuration

Location: `.claude/settings.local.json`

Available hooks:
- `SessionStart`: Runs when Claude Code starts
- `PostToolUse`: Runs after tool execution (optional, not configured by default)
- `Stop`: Runs when Claude Code stops (future enhancement)

## 📈 Memory System Components

### 1. Session Tracking
- **Purpose**: Track Claude Code sessions from start to finish
- **Data**: Session metadata, tool usage, file interactions
- **Storage**: SQLite database + JSON session files

### 2. Context Building  
- **Purpose**: Create relevant context for Claude Code
- **Sources**: Recent sessions, project memory, git info, decisions
- **Output**: Formatted markdown context injected into Claude

### 3. Project Memory
- **Purpose**: Persistent project knowledge and decisions
- **Format**: Structured markdown with sections for different types of information
- **Location**: `.prsist/context/project_memory.md`

### 4. Decision Tracking
- **Purpose**: Record and track important project decisions
- **Categories**: Architecture, technical, process, design
- **Impact Levels**: Low, medium, high, critical

## 🔍 Monitoring and Debugging

### System Health Checks

```bash
# Complete system validation
python .prsist/test_system.py

# Check database connectivity
python -c "from database import MemoryDatabase; print('DB OK' if MemoryDatabase().get_recent_sessions(1) else 'DB Error')"

# Validate configuration
python -c "from memory_manager import MemoryManager; print(MemoryManager().validate_system())"
```

### Session Inspection

```bash
# View current session status
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
info = mm.get_session_info()
print(f'Active Session: {info.get(\"session_id\", \"None\")}')
print(f'Duration: {info.get(\"duration_minutes\", 0)} minutes')
print(f'Tools Used: {info.get(\"tool_usage_count\", 0)}')
"

# Review recent sessions
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
for session in mm.get_recent_sessions(5):
    print(f'{session[\"start_time\"]} - {session.get(\"summary\", \"No summary\")}')
"
```

### Context Analysis

```bash
# View what context Claude sees
python -c "
from memory_manager import MemoryManager
context = MemoryManager().get_session_context()
print(f'Context Length: {len(context)} characters')
print('\\n--- Context Preview ---')
print(context[:500] + '...' if len(context) > 500 else context)
"
```

## 🛠️ Maintenance

### Regular Maintenance Tasks

1. **Weekly System Health Check**
   ```bash
   python .prsist/test_system.py
   ```

2. **Monthly Data Cleanup**
   ```bash
   python -c "from memory_manager import MemoryManager; print(MemoryManager().cleanup_old_data(30))"
   ```

3. **Quarterly Configuration Review**
   - Review `.prsist/config/memory-config.yaml`
   - Check retention policies
   - Analyze memory usage patterns

### Backup and Recovery

```bash
# Backup entire memory system
cp -r .prsist/ .prsist-backup-$(date +%Y%m%d)

# Export active session data
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()
data = mm.export_session_data()
if data:
    with open('session_backup.json', 'w') as f:
        f.write(data)
"

# Database backup
cp .prsist/storage/sessions.db .prsist/storage/sessions-backup-$(date +%Y%m%d).db
```

## 🎯 Best Practices

### Feature Completion Logging

**Good Examples:**
```bash
# Specific and descriptive
python .prsist/hooks/FeatureComplete.py "User Authentication API" "OAuth2 implementation with JWT, refresh tokens, and rate limiting"

# Include impact and context
python .prsist/hooks/FeatureComplete.py "Database Migration" "Added user preferences schema, migrated 10K+ existing users successfully"

# Bug fixes with root cause
python .prsist/hooks/FeatureComplete.py "Memory Leak Fix" "Fixed session cleanup in authentication module, reduced memory usage by 40%"
```

**Avoid:**
```bash
# Too vague
python .prsist/hooks/FeatureComplete.py "Work done"

# Too technical without context  
python .prsist/hooks/FeatureComplete.py "Fixed bug in line 247"
```

### Decision Recording

**Structure decisions with:**
- Clear title describing what was decided
- Context explaining why the decision was needed
- Options considered and rationale for choice
- Impact assessment and future implications

### Context Management

- Keep project memory focused and relevant
- Archive old decisions that are no longer applicable
- Regular cleanup of obsolete session data
- Monitor context length to stay within limits

## 🔧 Troubleshooting

### Common Issues

1. **Hook Not Executing**
   - Check `.claude/settings.local.json` syntax
   - Verify Python executable in PATH
   - Test hook directly: `python .prsist/hooks/SessionStart.py`

2. **Database Connection Issues**
   - Check file permissions on `.prsist/storage/`
   - Verify SQLite installation
   - Run system validation: `python .prsist/test_system.py`

3. **Context Not Loading**
   - Check session tracking is working
   - Verify context builder configuration
   - Test context generation: `python -c "from memory_manager import MemoryManager; print(len(MemoryManager().get_session_context()))"`

4. **Performance Issues**
   - Check database size and optimize if needed
   - Adjust retention policies in configuration
   - Monitor hook execution time (should be < 2 seconds)

### Debug Mode

Enable detailed logging:
```bash
# Set environment variable for debug logging
export BMAD_MEMORY_DEBUG=1
python .prsist/hooks/SessionStart.py
```

## 🔮 Future Enhancements

### Phase 2 Features (Planned)
- Semantic similarity scoring for context relevance
- Advanced summarization of session data
- Cross-session learning and pattern recognition
- Integration with BMAD agent workflows

### Possible Integrations
- Git commit message generation from session history
- Automated documentation updates
- Code review context injection
- Project health monitoring

## 📞 Support and Community

### Getting Help
1. Run system diagnostics: `python .prsist/test_system.py`
2. Check configuration: `python -c "from memory_manager import MemoryManager; print(MemoryManager().validate_system())"`
3. Review recent logs for error messages
4. Export session data for analysis if needed

This integration guide provides everything needed to effectively use the Prsist Memory System. The system is designed to be invisible during normal development while providing powerful memory capabilities across sessions.