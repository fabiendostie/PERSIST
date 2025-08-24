# Prsist Memory System - JavaScript Bridge

## Overview

The Prsist JavaScript Bridge provides a Node.js interface to the Python-based Prsist Memory System, enabling integration with JavaScript workflows like BMAD-METHOD while maintaining workflow-agnostic portability.

## Architecture

```
┌─────────────────────────┐    ┌─────────────────────────┐
│   JavaScript/Node.js    │    │     Python Core         │
│                         │    │                         │
│  ┌─────────────────┐   │    │  ┌─────────────────┐   │
│  │   CLI Interface │   │    │  │ Memory Manager  │   │
│  └─────────────────┘   │    │  └─────────────────┘   │
│                         │    │                         │
│  ┌─────────────────┐   │◄──►│  ┌─────────────────┐   │
│  │ BMAD Adapter    │   │    │  │   Database      │   │
│  └─────────────────┘   │    │  └─────────────────┘   │
│                         │    │                         │
│  ┌─────────────────┐   │    │  ┌─────────────────┐   │
│  │ Bridge Core     │   │    │  │ Session Tracker │   │
│  └─────────────────┘   │    │  └─────────────────┘   │
└─────────────────────────┘    └─────────────────────────┘
```

## Key Features

### ✅ **Universal CLI Interface**
- Single command interface: `prsist <command>`
- Auto-detects workflow type (BMAD, generic, etc.)
- Works from any directory
- Chainable commands for efficiency

### ✅ **BMAD-Aware Integration**
- Captures agent decisions automatically
- Tracks story events and architecture decisions
- Provides agent-specific context
- Maintains cross-agent learning

### ✅ **Workflow-Agnostic Design**
- Core functionality works with any development workflow
- Plugin architecture for specific integrations
- No vendor lock-in
- Easy to extend for new workflows

### ✅ **Dual-Language Bridge**
- JavaScript ecosystem reach (NPM, Node.js tools)
- Python AI/ML capabilities (embeddings, analysis)
- Efficient inter-process communication
- JSON-based data exchange

## Installation

### Quick Setup
```bash
# From your project directory
npm install prsist-memory

# Or use directly with npx
npx prsist-memory status
```

### Development Setup
```bash
# Clone the bridge
git clone <repo-url>
cd prsist-bridge

# Install dependencies (none currently)
npm install

# Test the bridge
npm test
```

## Usage

### Basic Commands

```bash
# Check system status
prsist status

# Health check
prsist health

# View current context
prsist context

# Add to project memory
prsist memory "Decided to use React for frontend"

# Record decision
prsist decision "Use PostgreSQL" "Better ACID compliance" "high"

# Create checkpoint
prsist checkpoint "auth-system-complete" "Authentication system fully implemented"

# Session management
prsist start '{"project": "MyApp", "feature": "Authentication"}'
prsist end
```

### BMAD Workflow Commands

When in a BMAD project (auto-detected), additional commands become available:

```bash
# Capture agent decisions
prsist agent decision analyst "Use microservices architecture"
prsist agent decision architect "Implement API Gateway pattern"
prsist agent decision dev "Use TypeScript for type safety"

# Track story events
prsist story event "User Registration API" "created"
prsist story event "User Registration API" "started"
prsist story event "User Registration API" "completed"

# Architecture decisions
prsist arch "auth-service" "JWT token implementation" "Stateless auth preferred"

# Get agent-specific context
prsist agent context dev
```

### Advanced Usage

```bash
# Chain operations (efficient for scripts)
prsist status && prsist memory "Feature complete" && prsist checkpoint "milestone-1"

# Export session data
prsist stats

# Recent sessions
prsist recent 5

# Full system validation
prsist health
```

## API Usage

### Basic Integration

```javascript
const { createPrsistAdapter } = require('prsist-memory');

// Auto-detect workflow
const memory = createPrsistAdapter('auto');
await memory.initialize();

// Basic operations
await memory.startSession({ project: 'MyApp' });
await memory.addProjectMemory('Important architectural decision made');
await memory.createCheckpoint('feature-complete');
```

### BMAD Integration

```javascript
const { BmadPrsistAdapter } = require('prsist-memory');

const bmadMemory = new BmadPrsistAdapter();
await bmadMemory.initialize();

// Capture agent activities
await bmadMemory.captureAgentDecision(
    'analyst', 
    'Use microservices for better scalability',
    { confidence: 0.9, alternatives: ['monolith', 'modular-monolith'] }
);

// Track stories
await bmadMemory.captureStoryEvent(
    'User Authentication',
    'completed',
    { story_points: 8, developer: 'ai-agent' }
);

// Get context for next agent
const context = await bmadMemory.getBmadContext('dev');
```

### Workflow Detection

The bridge automatically detects workflow types:

- **BMAD**: Detects `bmad-method` in package.json or `bmad-core/` directory
- **Generic**: Fallback for any other workflow
- **Future**: Easy to add new workflow adapters

## Configuration

### Workflow Configuration

```javascript
// Explicit workflow specification
const memory = createPrsistAdapter('bmad', {
    pythonPath: 'python3',  // Custom Python path
    debug: true,            // Enable debug logging
    prsistRoot: '/custom/path'  // Custom Prsist location
});
```

### Environment Variables

```bash
export PRSIST_PYTHON_PATH=python3
export PRSIST_DEBUG=true
export PRSIST_ROOT=/path/to/prsist
```

## Integration Examples

### Git Hooks

```bash
#!/bin/bash
# .git/hooks/post-commit

# Correlate commit with current session
prsist memory "Committed: $(git log -1 --pretty=format:'%s')"
prsist checkpoint "commit-$(git rev-parse --short HEAD)"
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Track deployment
  run: |
    prsist start '{"deployment": "production", "version": "${{ github.sha }}"}'
    prsist memory "Deployed version ${{ github.sha }} to production"
    prsist checkpoint "production-deployment"
    prsist end
```

### BMAD Agent Hooks

```javascript
// In BMAD agent completion hook
const { BmadPrsistAdapter } = require('prsist-memory');

async function onAgentCompletion(agent, result) {
    const memory = new BmadPrsistAdapter();
    await memory.captureAgentDecision(agent.name, result.decision, result.context);
}
```

## Development

### Adding New Workflow Adapters

1. Extend the base `PrsistBridge` class
2. Add workflow-specific methods
3. Update the factory function
4. Add detection logic

```javascript
class MyWorkflowAdapter extends PrsistBridge {
    constructor(options = {}) {
        super({ ...options, workflowType: 'my-workflow' });
    }
    
    async captureMyWorkflowEvent(data) {
        return this.captureWorkflowEvent('my-event', data);
    }
}
```

### Testing

```bash
# Run all tests
npm test

# Test specific functionality
node prsist-bridge.js test
node cli.js health
node examples/bmad-integration.js
```

## Contributing

1. **Add workflow adapters** for new development frameworks
2. **Enhance CLI commands** for better developer experience  
3. **Improve auto-detection** for more workflow types
4. **Add integration examples** for popular tools

## License

MIT License - Same as the core Prsist Memory System.

## Links

- **Main Repository**: [Prsist Memory System](../README.md)
- **Python Core Documentation**: [../docs/](../docs/)
- **BMAD-METHOD**: [../../README.md](../../README.md)
- **NPM Package**: `prsist-memory` (when published)

---

**🚀 Ready for Phase 2-4 Implementation!**

This bridge enables the efficient implementation of Phases 2-4 by providing:
- **Immediate JavaScript ecosystem integration**
- **BMAD workflow awareness**  
- **Foundation for semantic analysis and AI features**
- **Open source distribution pathway**