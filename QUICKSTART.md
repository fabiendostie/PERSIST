# Prsist Memory System - Quick Start Guide

Get up and running with the Prsist Memory System in under 5 minutes. This guide will walk you through installation, setup, and your first AI session with persistent memory.

## Prerequisites

Before you begin, ensure you have:
- Python 3.7+ (Python 3.10+ recommended)
- Claude Code installed and configured
- Git (optional, but recommended for enhanced features)

## Installation

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/fabiendostie/PERSIST.git
cd PERSIST

# Run the automated installer
python install-prsist.py
```

The installer will:
- Set up the `.prsist/` system in your project
- Configure Claude Code integration
- Install necessary CLI tools
- Initialize the database
- Run a health check

### Option 2: Quick Distribution Install

```bash
# Download and run the portable installer
curl -sSL https://github.com/fabiendostie/PERSIST/releases/latest/install.py | python
```

## First Steps

### 1. Verify Installation

```bash
# Check system health
python .prsist/bin/prsist.py -h

# Expected output:
# [HEALTH] Health Check...
# [PASS] System healthy
```

### 2. Initialize Your First Session

```bash
# Check system status
python mem.py status

# View current context (should be minimal for new installation)
python mem.py context

# Check memory statistics
python mem.py memory
```

### 3. Test Claude Code Integration

Start a Claude Code session in your project directory. The Prsist system will automatically:
- Detect the new session
- Create a session record
- Begin tracking your interactions
- Build context as you work

## Basic Usage

### Essential Commands

**System Status and Health:**
```bash
python mem.py status          # Overall system status
python mem.py health          # Comprehensive health check
python mem.py memory          # Memory and performance statistics
```

**Context Management:**
```bash
python mem.py context         # View current context
python mem.py recent          # Show recent sessions
python mem.py export          # Export session data
```

**Project Memory:**
```bash
# Add a feature completion
python mem.py feature "User Authentication" "Implemented OAuth2 login system"

# Log a project decision
python mem.py decision "Chose PostgreSQL over MongoDB for better ACID compliance"

# Create a checkpoint
python mem.py checkpoint "v0.0.3 Release Candidate"
```

### Claude Code Slash Commands

Inside your Claude Code sessions, use these commands:

```bash
/mem-status                   # Quick system status
/mem-context                  # Show current context
/mem-memory                   # Memory statistics
/mem-recent                   # Recent sessions
/mem-feature <name> <desc>    # Log feature completion
/mem-decision <decision>      # Log project decision
/mem-checkpoint <name>        # Create checkpoint
/mem-validate                 # System validation
```

## Understanding Your Memory System

### What Gets Tracked

The Prsist system automatically tracks:
- **Session Activities**: All tool usage, file operations, and conversations
- **Code Changes**: Git integration tracks repository changes
- **Project Context**: Important files, recent changes, and project state
- **Decisions**: Major decisions and their rationale
- **Features**: Completed features and milestones
- **Performance**: System health and performance metrics

### Where Your Data Lives

```
.prsist/
├── storage/
│   ├── sessions.db          # Main database with all session data
│   └── memory.log          # Activity log
├── context/
│   ├── claude-context.md   # Current session context
│   └── project-memory.md   # Long-term project memory
└── sessions/
    ├── active/            # Current session data
    └── archived/          # Historical sessions
```

### Memory in Action

1. **Start a Claude Code session** - Prsist automatically creates a session record
2. **Work normally** - All tool usage is tracked transparently
3. **Context builds** - Relevant context accumulates as you work
4. **Next session** - Previous context is automatically loaded
5. **Continuity** - Pick up exactly where you left off

## Advanced Features

### Semantic Analysis

The system includes AI-powered semantic analysis:
- **Code Understanding**: Analyzes code changes for meaning, not just syntax
- **Context Relevance**: Ranks context by relevance to current work
- **Pattern Recognition**: Identifies development patterns and best practices

### Performance Optimization

- **KV-Cache**: Improved context retrieval performance
- **Smart Filtering**: Reduced noise while maintaining relevance
- **Token Optimization**: Efficient use of AI model token limits

### Cross-Session Intelligence

- **Work Correlation**: Links related work across different sessions
- **Decision Tracking**: Maintains decision history across time
- **Feature Evolution**: Tracks how features develop over time

## Common Workflows

### Daily Development

```bash
# Start your day - check what you were working on
python mem.py context
python mem.py recent

# Start Claude Code session
# ... work normally ...

# End of day - log what you completed
python mem.py feature "Bug Fix" "Fixed critical authentication issue"
```

### Project Milestones

```bash
# Before a release
python mem.py checkpoint "Release v0.0.3"

# After a major decision
python mem.py decision "Migrated from REST to GraphQL for better API flexibility"
```

### Debugging and Analysis

```bash
# Find related sessions
python mem.py recent | grep "authentication"

# Export for analysis
python mem.py export > project_history.json

# Check system performance
python mem.py health
```

## Troubleshooting

### Common Issues

**"System not found" or "Health check failed":**
```bash
# Ensure you're in the project root directory
cd /path/to/your/project

# Re-run initialization
python .prsist/bin/prsist.py --init
```

**"Context not loading in Claude Code":**
```bash
# Check context status
python mem.py context

# Validate system
python mem.py health

# Check Claude Code integration
ls .claude/commands/
```

**"Performance issues":**
```bash
# Check system health
python mem.py health

# View performance metrics
python mem.py memory

# Clear old sessions (if database is large)
python .prsist/bin/prsist.py --cleanup
```

### Debug Mode

Enable detailed logging by editing `.prsist/config/memory-config.yaml`:

```yaml
debug: true
log_level: DEBUG
```

## Next Steps

### Explore Advanced Features

- Review `FEATURES.md` for comprehensive feature documentation
- Explore the `/mem-*` commands in Claude Code
- Set up git hooks for enhanced integration
- Configure performance monitoring

### Customize Your Setup

- Adjust context filtering in `.prsist/config/memory-config.yaml`
- Add custom slash commands in `.claude/commands/`
- Set up automated checkpoints for your workflow
- Configure export formats for your analysis needs

### Getting Help

- Check the main `README.md` for detailed documentation
- Run `python .prsist/bin/prsist.py --help` for CLI help
- Use `/mem-validate` in Claude Code to check system integrity
- Review logs in `.prsist/storage/memory.log` for detailed information

## What's Next?

Now that you have Prsist running, you'll experience:
- **Seamless continuity** across Claude Code sessions
- **Intelligent context** that improves over time
- **Project memory** that preserves important decisions and milestones
- **Performance insights** into your development patterns

The system learns from your usage patterns and becomes more helpful over time. The more you use it, the better it becomes at providing relevant context and insights.

---

**Need help?** Check the troubleshooting section above or review the comprehensive documentation in `README.md` and `FEATURES.md`.