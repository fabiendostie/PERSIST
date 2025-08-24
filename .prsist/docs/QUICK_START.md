# Prsist Memory System - Quick Start Guide

## ✅ System Already Installed and Working!

The memory system is **already active** and working in your Claude Code sessions.

## 🎯 **Verify It's Working**

```bash
# Test the complete system (should show all tests passing)
python .prsist/test_system.py

# Check if memory is being loaded (should return context about the project)
python .prsist/hooks/SessionStart.py
```

## 📋 **Daily Usage**

### Automatic Features (No Action Required)
- ✅ **Session tracking** when Claude Code starts/stops  
- ✅ **Context loading** from previous sessions
- ✅ **Project memory** maintained across sessions

### Manual Commands (Use When Needed)

## 🚀 **Simple Commands (New!)**

### **Easy-to-Remember CLI**
```bash
# Single command interface - much easier!
python .prsist/prsist.py -h    # Health check
python .prsist/prsist.py -t    # Test system  
python .prsist/prsist.py -s    # Session status
python .prsist/prsist.py -c    # Context (what Claude sees)
python .prsist/prsist.py -m    # Memory stats
python .prsist/prsist.py -r    # Recent sessions
python .prsist/prsist.py -f    # Feature log (interactive)

# Chain multiple commands together!
python .prsist/prsist.py -hm   # Health + Memory stats
python .prsist/prsist.py -tsc  # Test + Status + Context  
python .prsist/prsist.py -a    # All checks

# List all available commands
python .prsist/prsist.py -l    # Help
```

### **Complete Command Reference**

#### **Core Operations**
| Flag | Command | Description |
|------|---------|-------------|
| `-t` | Test | Complete system test |
| `-s` | Status | Current session info |
| `-c` | Context | What Claude sees |
| `-r` | Recent | Recent sessions |
| `-h` | Health | System health check |
| `-m` | Memory | Memory statistics |
| `-v` | Validate | System validation |

#### **Session Management**
| Flag | Command | Description |
|------|---------|-------------|
| `-n` | New | Start new session |
| `-e` | End | End current session |
| `-k` | Checkpoint | Create checkpoint |
| `-x` | Export | Export session data |

#### **Data Management**
| Flag | Command | Description |
|------|---------|-------------|
| `-f` | Feature | Log feature completion |
| `-p` | Project | Add to project memory |
| `-d` | Decision | Record decision |
| `-z` | Cleanup | Clean old data |

#### **Shortcuts**
| Flag | Command | Description |
|------|---------|-------------|
| `-a` | All | All core checks |
| `-l` | List | Show all commands |

#### Log Feature Completions (Interactive)
```bash
# Interactive feature logging
python .prsist/prsist.py -f
# Will prompt for feature name and description
```

#### Advanced: Original Commands (Still Available)
```bash
# Direct feature logging (non-interactive)
python .prsist/hooks/FeatureComplete.py "Feature Name" "Description"

# Direct Python commands (for scripts)
python -c "from memory_manager import MemoryManager; print(MemoryManager().get_session_info())"
```

#### Add Project Information  
```bash
# Add important project information manually
python -c "from memory_manager import MemoryManager; MemoryManager().add_project_memory('Important decision: Using SQLite for session storage due to performance requirements')"

# Record an architectural decision
python -c "
from memory_manager import MemoryManager
mm = MemoryManager()  
mm.add_decision(
    title='API Rate Limiting Strategy',
    description='Implemented sliding window rate limiting with Redis backend for better performance',
    category='architecture',
    impact='medium'
)
print('Decision recorded')
"
```

## 🔍 **What the System Remembers**

The memory system automatically tracks and remembers:

1. **Session History** - Previous Claude Code sessions and their outcomes
2. **Tool Usage** - What tools were used and their results  
3. **Project Decisions** - Important architectural and technical decisions
4. **Feature Milestones** - Completed features and their descriptions
5. **Git Information** - Current branch, commit status, project structure
6. **File Interactions** - Which files were modified and when

## 📊 **System Health Monitoring**

### Quick Health Check
```bash
# Complete system validation (should show "All tests passed!")
python .prsist/test_system.py
```

### Detailed System Info
```bash
# Validate system integrity
python -c "from memory_manager import MemoryManager; validation = MemoryManager().validate_system(); print('System Status:', 'OK' if validation['valid'] else 'ISSUES'); print('Issues:', validation.get('issues', []) if not validation['valid'] else 'None')"

# Get memory system statistics
python -c "from memory_manager import MemoryManager; stats = MemoryManager().get_memory_stats(); print(f'Total sessions: {stats.get(\"total_sessions\", 0)}'); print(f'Database size: {stats.get(\"database_size_mb\", 0)} MB')"
```

## 🛠️ **Troubleshooting**

### Common Issues

**Hook not executing:**
```bash
# Check Claude Code settings
cat .claude/settings.local.json | grep -A 5 hooks

# Test hook directly
python .prsist/hooks/SessionStart.py
```

**Database issues:**
```bash
# Test database connection
python -c "from database import MemoryDatabase; db = MemoryDatabase(); sessions = db.get_recent_sessions(1); print(f'Database OK: Found {len(sessions)} sessions')"
```

**Memory not loading:**
```bash
# Check if session tracking is working
python -c "from memory_manager import MemoryManager; mm = MemoryManager(); session_info = mm.get_session_info(); print(f'Active session: {session_info.get(\"session_id\", \"None\")}')"
```

## 📁 **File Structure**

```
.prsist/
├── docs/                           # Documentation (this guide)
├── config/memory-config.yaml       # System configuration  
├── storage/sessions.db             # SQLite database
├── context/project-memory.md       # Project knowledge
├── hooks/                          # Claude Code integration
│   ├── SessionStart.py             # Session initialization 
│   └── FeatureComplete.py          # Feature logging
├── sessions/                       # Session data
│   ├── active/                     # Current session
│   └── archived/                   # Past sessions
└── [system modules...]             # Core implementation
```

## 🎯 **Success Indicators**

You'll know the system is working when:

1. ✅ Claude Code sessions start with relevant project context
2. ✅ Claude remembers previous conversations and decisions  
3. ✅ Feature completions are logged and create checkpoints
4. ✅ Project memory grows with important information
5. ✅ System tests pass consistently

## 📚 **Additional Documentation**

- `MANUAL_TRIGGERS.md` - Complete command reference  
- `INTEGRATION_GUIDE.md` - Detailed integration information
- `SYSTEM_STATUS.md` - Current implementation status

## ✨ **Key Benefit**

**No more starting from scratch!** Each Claude Code session now begins with full context about:
- What was accomplished previously
- Current project status and decisions  
- Recent development patterns and insights
- Relevant historical context for the current task

The memory system provides true continuity across development sessions.