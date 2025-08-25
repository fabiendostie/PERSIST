# Prsist Memory System

Persistent memory system for AI conversations, enabling Claude Code and other AI assistants to maintain context across sessions.

## Overview

Prsist solves the fundamental problem of AI memory loss between sessions. It provides automatic session tracking, context injection, and project memory management, ensuring continuity in development workflows.

## Key Features

### Core Functionality
- **Persistent Session Memory**: Maintains conversation history and context across Claude Code sessions
- **Automatic Context Injection**: Loads relevant project history when sessions start
- **Project Memory**: Stores long-term decisions, features, and project knowledge
- **Tool Usage Tracking**: Records all tool interactions for complete activity history
- **Git Memory Management**: Automatic pause/resume of memory system during git operations
- **Safe Git Operations**: Force-add memory files and prevent conflicts during commits/merges
- **SQLite Backend**: Efficient, portable database storage for all memory data

### Claude Code Integration
- **Slash Commands**: 17+ custom commands for memory management (`/mem-status`, `/mem-context`, etc.)
- **Automatic Hooks**: Transparent operation through PostToolUse hooks
- **Memory Agent**: Dedicated agent for memory-related tasks
- **Session Management**: Automatic session creation, tracking, and archival

### Developer Tools
- **CLI Access**: Multiple interfaces (mem.py, memory-cli.py, claude-commands.py)
- **Health Monitoring**: Built-in system health checks and validation
- **Export Capabilities**: Session data export for analysis and backup
- **Checkpoint System**: Feature milestone tracking and rollback support

## Installation

### Quick Install (Recommended)

```bash
# Download and run installer
curl -sSL https://github.com/fabiendostie/PERSIST/releases/latest/install.py | python

# Or clone and install
git clone https://github.com/fabiendostie/PERSIST.git
cd PERSIST
python install-prsist.py
```

### Manual Installation

1. Copy `.prsist/` folder to your project root
2. Copy `.claude/` integration files to project root
3. Copy CLI scripts (`mem.py`, `memory-cli.py`, `claude-commands.py`)
4. Copy `.lefthook.yml` for git integration
5. Run initialization: `python .prsist/bin/prsist.py --init`

## Usage

### Command Line Interface

```bash
# System commands
python mem.py status          # Check system status
python mem.py health          # Run health check
python mem.py context         # Show current context
python mem.py memory          # Memory statistics
python mem.py recent          # Recent sessions

# Feature tracking
python mem.py feature "API Integration" "Completed REST API endpoints"

# Decision logging
python mem.py decision "Use PostgreSQL for better performance"

# Git memory management
python .prsist/bin/git-memory-manager.py status
python .prsist/bin/git-memory-manager.py commit "Safe commit with memory management"

# Force context injection (failsafe)
python .prsist/bin/prsist.py -i
```

### Claude Code Commands

```bash
/mem-status                   # System status and health
/mem-context                  # Current context information
/mem-memory                   # Memory statistics
/mem-feature                  # Log completed feature
/mem-decision                 # Add project decision
/mem-checkpoint               # Create manual checkpoint
/mem-recent                   # Show recent sessions
/mem-project-memory          # Add to persistent project memory
/mem-export                   # Export session data
/mem-validate                 # Validate system integrity
/mem-force-context            # Force context injection (failsafe)
```

### Advanced Usage

```bash
# Direct prsist CLI
python .prsist/bin/prsist.py -h     # Help
python .prsist/bin/prsist.py -s     # Status
python .prsist/bin/prsist.py -c     # Context
python .prsist/bin/prsist.py -k     # Checkpoint
python .prsist/bin/prsist.py -p     # Project memory

# Bridge for Node.js integration
node .prsist/bridge/cli.js status
node .prsist/bridge/prsist-bridge.js
```

## Architecture

### System Components

```
.prsist/
├── bin/                      # Core executables
│   ├── prsist.py            # Main CLI interface
│   ├── git-memory-manager.py # Git memory management
│   └── claude-integration.py # Claude Code integration
├── hooks/                    # Event hooks
│   ├── PostToolUse.py       # Tool usage tracking
│   └── FeatureComplete.py   # Feature milestone logging
├── config/                   # Configuration files
│   ├── memory-config.yaml   # System configuration
│   └── session-start.json   # Session templates
├── storage/                  # Data persistence
│   ├── sessions.db          # SQLite database
│   └── memory.log           # Activity log
├── context/                  # Context management
│   ├── claude-context.md    # Active context
│   └── project-memory.md    # Persistent memory
└── sessions/                 # Session data
    ├── active/              # Current session
    ├── archived/            # Historical sessions
    └── checkpoints/         # Milestone snapshots
```

### Key Modules

- **memory_manager.py**: Core memory management and API
- **session_tracker.py**: Session lifecycle management
- **context_builder.py**: Context generation and filtering
- **database.py**: SQLite database interface
- **utils.py**: Shared utilities and helpers

## Current Implementation Status

### Fully Implemented ✅
- Core memory system with SQLite backend
- Claude Code integration via hooks and commands
- Session tracking and management
- Context injection and filtering
- Project memory persistence
- Git memory management with automatic pause/resume hooks
- Safe git operations and conflict prevention
- CLI tools and interfaces
- Export and checkpoint functionality
- Health monitoring and validation
- Portable installer system

### In Development 🚧
- Web UI dashboard
- Advanced analytics
- Multi-user support
- Cloud sync capabilities
- Extended IDE integrations

## Distribution

### Creating a Distribution Package

```bash
# Create distribution
python create-distribution.py

# Create with ZIP archive
python create-distribution.py --zip
```

### Distribution Contents
- Complete `.prsist/` system
- Claude Code integration files
- CLI convenience scripts
- Installation and configuration
- Comprehensive documentation

## System Requirements

- Python 3.7+
- SQLite3 (included with Python)
- Git (optional, for hook integration)
- Claude Code (for AI integration features)

## Performance

- Hook execution: < 2 seconds
- Memory usage: < 50MB
- Database operations: Atomic transactions
- Context size: Optimized for token limits

## Security

- Path validation on all file operations
- SQL injection prevention via parameterized queries
- Graceful error handling and recovery
- No credential storage in memory files

## Troubleshooting

### Common Issues

1. **System not activating**
   ```bash
   python .prsist/bin/prsist.py -h  # Run health check
   ```

2. **Context not loading**
   ```bash
   python .prsist/bin/prsist.py -c  # Check context
   ```

3. **Database errors**
   ```bash
   python .prsist/bin/prsist.py -v  # Validate system
   ```

### Debug Mode

Enable debug logging in `.prsist/config/memory-config.yaml`:
```yaml
debug: true
```

## Contributing

Contributions welcome! The project uses:
- Conventional commits for version control
- Python type hints for code clarity
- Comprehensive error handling
- Modular architecture for extensibility

## License

MIT License - See LICENSE file for details

## Author

Fabien Dostie

## Acknowledgments

Built for and tested with Claude Code (Anthropic) to solve the persistent memory challenge in AI-assisted development.

---

**Project Status**: Active Development | **Version**: 0.0.1 | **Last Updated**: 2025-08-24