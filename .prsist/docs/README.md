# Prsist Memory System

Advanced persistent memory system for AI conversations with enhanced semantic analysis, git integration, and productivity tracking. Enables Claude Code and other AI assistants to maintain context across sessions with intelligent correlation and insights.

## Overview

Prsist solves the fundamental problem of AI memory loss between sessions while providing advanced analytics and insights. It features automatic session tracking, enhanced semantic analysis with TF-IDF, git integration, productivity measurement, and cross-session correlation for comprehensive development workflow support.

## Key Features

### Core Functionality
- **Persistent Session Memory**: Maintains conversation history and context across Claude Code sessions
- **Enhanced Semantic Analysis**: TF-IDF-powered text analysis with scikit-learn for intelligent similarity scoring
- **Automatic Context Injection**: Loads relevant project history when sessions start with semantic relevance
- **Project Memory**: Stores long-term decisions, features, and project knowledge with enhanced search
- **Tool Usage Tracking**: Records all tool interactions for complete activity history and analysis
- **SQLite Backend**: Efficient, portable database storage with extended schema for analytics

### Advanced Phase 2-3 Features
- **Enhanced Git Integration**: Automatic commit correlation, branch context tracking, merge analysis
- **Productivity Tracking**: Development velocity measurement, pattern analysis, effectiveness metrics  
- **Cross-Session Correlation**: Multi-dimensional analysis correlating sessions by time, semantics, git state, and behavior
- **Semantic Code Analysis**: Language-aware code parsing with embedding generation for Python, JavaScript, Java, C++
- **TF-IDF Vectorization**: Advanced text similarity using scikit-learn with cosine similarity calculations
- **Git Hook Management**: Automated git hooks for seamless workflow integration

### Claude Code Integration
- **Slash Commands**: 20+ custom commands for memory management (`/mem-status`, `/mem-context`, `/mem-git-report`, etc.)
- **Automatic Hooks**: Transparent operation through PostToolUse hooks with enhanced correlation
- **Memory Agent**: Dedicated agent for memory-related tasks with advanced features
- **Session Management**: Automatic session creation, tracking, archival, and correlation analysis

### Developer Tools
- **CLI Access**: Multiple interfaces (mem.py, memory-cli.py, claude-commands.py) with git integration
- **Health Monitoring**: Built-in system health checks and validation with dependency verification
- **Export Capabilities**: Session data export for analysis and backup with correlation data
- **Checkpoint System**: Feature milestone tracking and rollback support with git integration
- **Productivity Insights**: Development velocity reports and pattern analysis

## Installation

### Dependencies (Phase 2-3 Enhanced Features)

For full functionality with TF-IDF semantic analysis:

```bash
# Install required Python packages
pip install numpy scikit-learn scipy joblib threadpoolctl

# Or install via provided scripts (handles compatibility issues)
python packages/install_wheels.py
```

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
```

### Claude Code Commands

```bash
# Core Memory Commands
/mem-status                   # System status and health
/mem-context                  # Current context information
/mem-memory                   # Memory statistics
/mem-recent                   # Show recent sessions
/mem-project-memory          # Add to persistent project memory
/mem-export                   # Export session data
/mem-validate                 # Validate system integrity

# Phase 2-3 Enhanced Commands
/mem-git-status              # Git integration status
/mem-git-report              # Comprehensive git correlation report
/mem-git-switch-branch       # Switch branches with memory correlation
/mem-productivity            # Development velocity and productivity metrics
/mem-correlation             # Cross-session correlation analysis
/mem-semantic                # Semantic analysis of current session

# Feature Tracking
/mem-feature                  # Log completed feature
/mem-decision                 # Add project decision
/mem-checkpoint               # Create manual checkpoint
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

#### Core System
- **memory_manager.py**: Core memory management and API with enhanced git integration
- **session_tracker.py**: Session lifecycle management
- **context_builder.py**: Context generation and filtering
- **database.py**: SQLite database interface with extended schema
- **utils.py**: Shared utilities and helpers

#### Phase 2-3 Enhanced Modules
- **enhanced_git_integration.py**: Advanced git integration with commit correlation and branch management
- **productivity_tracker.py**: Development velocity measurement and productivity analysis
- **semantic_analyzer.py**: TF-IDF semantic analysis and code parsing with multi-language support
- **cross_session_correlator.py**: Multi-dimensional correlation analysis engine

## Current Implementation Status

### Phase 1 - Foundation ✅
- Core memory system with SQLite backend
- Claude Code integration via hooks and commands
- Session tracking and management
- Context injection and filtering
- Project memory persistence
- Basic git integration via lefthook
- CLI tools and interfaces
- Export and checkpoint functionality
- Health monitoring and validation
- Portable installer system

### Phase 2-3 - Enhanced Analytics ✅ (NEW!)
- **Enhanced Git Integration**: Automatic commit correlation, branch context tracking, merge analysis
- **Productivity Tracking**: Development velocity measurement, pattern analysis, effectiveness metrics  
- **Advanced Semantic Analysis**: TF-IDF vectorization with scikit-learn, cosine similarity calculations
- **Cross-Session Correlation**: Multi-dimensional analysis (temporal, semantic, git state, behavioral)
- **Code Parsing Engine**: Multi-language support (Python, JavaScript, Java, C++) with AST analysis
- **Extended Database Schema**: 20+ new methods for analytics, git correlations, workflow events
- **Enhanced CLI Commands**: Git reporting, productivity metrics, correlation analysis
- **Automated Testing Suite**: Comprehensive test runner with performance benchmarks

### Dependencies Resolved ✅
- **NumPy 2.3.2**: Array operations and mathematical functions
- **SciPy 1.16.1**: Scientific computing libraries  
- **Scikit-learn 1.7.1**: TF-IDF vectorization and machine learning features
- **Joblib 1.5.1**: Parallel processing support
- **ThreadPoolCtl 3.6.0**: Thread pool management

### In Development 🚧
- Web UI dashboard with analytics visualization
- Cloud sync capabilities
- Extended IDE integrations
- Advanced machine learning features

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

### Core System
- Python 3.7+ (3.13+ recommended for enhanced features)
- SQLite3 (included with Python)
- Git (required for Phase 2-3 git integration)
- Claude Code (for AI integration features)

### Enhanced Features (Phase 2-3)
- **NumPy**: Array operations and mathematical computations
- **SciPy**: Scientific computing and spatial algorithms
- **Scikit-learn**: TF-IDF vectorization and machine learning
- **Joblib**: Parallel processing for enhanced performance
- **ThreadPoolCtl**: Thread pool management for scikit-learn

## Performance

### Core Operations
- Hook execution: < 2 seconds
- Memory usage: < 50MB base system
- Database operations: Atomic transactions
- Context size: Optimized for token limits

### Enhanced Features (Phase 2-3)
- Semantic analysis: < 15 seconds for large files
- Git correlation: < 10 seconds for session analysis  
- TF-IDF vectorization: < 5 seconds for typical text
- Cross-session correlation: < 30 seconds for 7-day analysis
- Productivity tracking: < 5 seconds for velocity measurement

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

**Project Status**: Phase 2-3 Complete | **Version**: 0.0.2 | **Last Updated**: 2025-08-24