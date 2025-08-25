# CLAUDE.md

Don't be an ass kisser, don't glaze my donut, keep it to the point. Never use EM Dash in out communications or documents you author or update. Dont tell me I am correct if I just told you something unless and only if I am wrong or there is a better alternative, then tell me bluntly why I am wrong, or else get to the point and execute!

## Markdown Linting Conventions

Always follow these markdown linting rules:

- **Blank lines around headings**: Always leave a blank line before and after headings
- **Blank lines around lists**: Always leave a blank line before and after lists
- **Blank lines around code fences**: Always leave a blank line before and after fenced code blocks
- **Fenced code block languages**: All fenced code blocks must specify a language (use `text` for plain text)
- **Single trailing newline**: Files should end with exactly one newline character
- **No trailing spaces**: Remove any trailing spaces at the end of lines

## BMAD-METHOD Overview

BMAD-METHOD is an AI-powered Agile development framework that provides specialized AI agents for software development. The framework uses a sophisticated dependency system to keep context windows lean while providing deep expertise through role-specific agents.

## Essential Commands

### Build and Validation

```bash
npm run build              # Build all web bundles (agents and teams)
npm run build:agents       # Build agent bundles only
npm run build:teams        # Build team bundles only
npm run validate           # Validate all configurations
npm run format             # Format all markdown files with prettier
```

### Development and Testing

```bash
npx bmad-build build                # Alternative build command via CLI
npx bmad-build list:agents          # List all available agents
npx bmad-build validate             # Validate agent configurations
```

### Installation Commands

```bash
npx bmad-method install             # Install stable release (recommended)
npx bmad-method@beta install        # Install bleeding edge version
npx bmad-method@latest install      # Explicit stable installation
npx bmad-method@latest update       # Update stable installation
npx bmad-method@beta update         # Update bleeding edge installation
```

### Dual Publishing Strategy

The project uses a dual publishing strategy with automated promotion:

**Branch Strategy:**
- `main` branch: Bleeding edge development, auto-publishes to `@beta` tag
- `stable` branch: Production releases, auto-publishes to `@latest` tag

**Release Promotion:**
1. **Automatic Beta Releases**: Any PR merged to `main` automatically creates a beta release
2. **Manual Stable Promotion**: Use GitHub Actions to promote beta to stable

**Promote Beta to Stable:**
1. Go to GitHub Actions tab in the repository
2. Select "Promote to Stable" workflow
3. Click "Run workflow"
4. Choose version bump type (patch/minor/major)
5. The workflow automatically:
   - Merges main to stable
   - Updates version numbers
   - Triggers stable release to NPM `@latest`

**User Experience:**
- `npx bmad-method install` → Gets stable production version
- `npx bmad-method@beta install` → Gets latest beta features
- Team develops on bleeding edge without affecting production users

### Release and Version Management

```bash
npm run version:patch      # Bump patch version
npm run version:minor      # Bump minor version
npm run version:major      # Bump major version
npm run release           # Semantic release (CI/CD)
npm run release:test      # Test release configuration
```

### Version Management for Core and Expansion Packs

#### Bump All Versions (Core + Expansion Packs)

```bash
npm run version:all:major   # Major version bump for core and all expansion packs
npm run version:all:minor   # Minor version bump for core and all expansion packs (default)
npm run version:all:patch   # Patch version bump for core and all expansion packs
npm run version:all         # Defaults to minor bump
```

#### Individual Version Bumps

For BMad Core only:
```bash
npm run version:core:major  # Major version bump for core only
npm run version:core:minor  # Minor version bump for core only
npm run version:core:patch  # Patch version bump for core only
npm run version:core        # Defaults to minor bump
```

For specific expansion packs:
```bash
npm run version:expansion bmad-creator-tools       # Minor bump (default)
npm run version:expansion bmad-creator-tools patch # Patch bump
npm run version:expansion bmad-creator-tools minor # Minor bump
npm run version:expansion bmad-creator-tools major # Major bump

# Set specific version (old method, still works)
npm run version:expansion:set bmad-creator-tools 2.0.0
```

## Architecture and Code Structure

### Core System Architecture

The framework uses a **dependency resolution system** where agents only load the resources they need:

1. **Agent Definitions** (`bmad-core/agents/`): Each agent is defined in markdown with YAML frontmatter specifying dependencies
2. **Dynamic Loading**: The build system (`tools/lib/dependency-resolver.js`) resolves and includes only required resources
3. **Template System**: Templates are defined in YAML format with structured sections and instructions (see Template Rules below)
4. **Workflow Engine**: YAML-based workflows in `bmad-core/workflows/` define step-by-step processes

### Key Components

- **CLI Tool** (`tools/cli.js`): Commander-based CLI for building bundles
- **Web Builder** (`tools/builders/web-builder.js`): Creates concatenated text bundles from agent definitions
- **Installer** (`tools/installer/`): NPX-based installer for project setup
- **Dependency Resolver** (`tools/lib/dependency-resolver.js`): Manages agent resource dependencies

### Build System

The build process:

1. Reads agent/team definitions from `bmad-core/`
2. Resolves dependencies using the dependency resolver
3. Creates concatenated text bundles in `dist/`
4. Validates configurations during build

### Critical Configuration

**`bmad-core/core-config.yaml`** is the heart of the framework configuration:

- Defines document locations and expected structure
- Specifies which files developers should always load
- Enables compatibility with different project structures (V3/V4)
- Controls debug logging

## Development Practices

### Adding New Features

1. **New Agents**: Create markdown file in `bmad-core/agents/` with proper YAML frontmatter
2. **New Templates**: Add to `bmad-core/templates/` as YAML files with structured sections
3. **New Workflows**: Create YAML in `bmad-core/workflows/`
4. **Update Dependencies**: Ensure `dependencies` field in agent frontmatter is accurate

### Important Patterns

- **Dependency Management**: Always specify minimal dependencies in agent frontmatter to keep context lean
- **Template Instructions**: Use YAML-based template structure (see Template Rules below)
- **File Naming**: Follow existing conventions (kebab-case for files, proper agent names in frontmatter)
- **Documentation**: Update user-facing docs in `docs/` when adding features

### Template Rules

Templates use the **BMad Document Template** format (`/Users/brianmadison/dev-bmc/BMAD-METHOD/common/utils/bmad-doc-template.md`) with YAML structure:

1. **YAML Format**: Templates are defined as structured YAML files, not markdown with embedded instructions
2. **Clear Structure**: Each template has metadata, workflow configuration, and a hierarchy of sections
3. **Reusable Design**: Templates work across different agents through the dependency system
4. **Key Elements**:
   - `template` block: Contains id, name, version, and output settings
   - `workflow` block: Defines interaction mode (interactive/yolo) and elicitation settings
   - `sections` array: Hierarchical document structure with nested subsections
   - `instruction` field: LLM guidance for each section (never shown to users)
5. **Advanced Features**:
   - Variable substitution: `{{variable_name}}` syntax for dynamic content
   - Conditional sections: `condition` field for optional content
   - Repeatable sections: `repeatable: true` for multiple instances
   - Agent permissions: `owner` and `editors` fields for access control
6. **Clean Output**: All processing instructions are in YAML fields, ensuring clean document generation

## Notes for Claude Code

- The project uses semantic versioning with automated releases via GitHub Actions
- All markdown is formatted with Prettier (run `npm run format`)
- Expansion packs in `expansion-packs/` provide domain-specific capabilities
- NEVER automatically commit or push changes unless explicitly asked by the user
- NEVER include Claude Code attribution or co-authorship in commit messages

## Prsist Memory System Integration

The Prsist Memory System is automatically active for Claude Code sessions. It provides:

- **Project Memory**: Persistent memory across conversations
- **Context Tracking**: Automatic context updates as you work  
- **Decision Logging**: Track important project decisions
- **Session Management**: Correlate work across sessions

### Commands for Claude

When needed, Claude can use these memory commands:

```bash
# Check memory status
python .prsist/bin/prsist.py -h

# View current context
python .prsist/bin/prsist.py -c

# Add project memory
python .prsist/bin/prsist.py -p

# Create checkpoint
python .prsist/bin/prsist.py -k
```

### Transparent Operation

The system runs transparently in the background:
- Auto-starts with Claude Code sessions
- Updates context after tool usage
- Maintains session history
- No user interaction required
### Recent Development Summary
We've been working on the Prsist Memory System v0.0.3, completing Phase 2-4 features including:
- ✅ Fixed all component initialization issues
- ✅ Installed AI dependencies (numpy, scikit-learn, sentence-transformers)
- ✅ All 15 components across phases now operational (100% success rate)
- ✅ Created 23 Claude Code slash commands
- ✅ Performance monitoring working (16.2MB memory usage)
- ✅ Documentation updated and corrected

### Recent Sessions
- **Session 21add851** (2025-08-25 07:21:17): 0 tools used
- **Session 171e8ac3** (2025-08-25 03:54:17): 0 tools used
- **Session ba33a8e2** (2025-08-25 03:41:55): 0 tools used

### What We Just Completed
- Fixed context injection bug in SessionStart.py
- Added missing slash commands: /mem-productivity, /mem-semantic, /mem-analytics, /mem-knowledge, /mem-optimize, /mem-correlate
- Corrected documentation version numbers from 2.0.0 to 0.0.3
- Verified all performance claims match actual test results

### Next Priority Tasks
- **TEST CONTEXT INJECTION**: Verify new sessions receive project context
- **VALIDATE MEMORY SYSTEM**: Ensure cross-session continuity works
- **PRODUCTION READINESS**: Final validation before deployment

### How to Test Memory System
- Start new Claude Code session in different terminal
- Ask "where were we in the implementation?"  
- Should receive context about Phase 2-4 completion
- Use `/mem-status` and `/mem-context` commands

---
## Current Session Context

**Last Updated:** 2025-08-25 03:23:36

### Recent Development Summary
We've been working on the Prsist Memory System v0.0.3, completing Phase 2-4 features including:
- Fixed all component initialization issues
- Installed AI dependencies (numpy, scikit-learn, sentence-transformers)
- All 15 components across phases now operational (100% success rate)
- Created 23 Claude Code slash commands  
- Performance monitoring working (16.2MB memory usage)
- Documentation updated and corrected

### Recent Sessions
- **Session 8389f7db** (2025-08-25 07:23:27): 0 tools used
- **Session 21add851** (2025-08-25 07:21:17): 0 tools used
- **Session 171e8ac3** (2025-08-25 03:54:17): 0 tools used

### What We Just Completed
- Fixed context injection bug in SessionStart.py
- Added missing slash commands: /mem-productivity, /mem-semantic, /mem-analytics, /mem-knowledge, /mem-optimize, /mem-correlate
- Corrected documentation version numbers from 2.0.0 to 0.0.3
- Verified all performance claims match actual test results

### Next Priority Tasks
- **TEST CONTEXT INJECTION**: Verify new sessions receive project context
- **VALIDATE MEMORY SYSTEM**: Ensure cross-session continuity works  
- **PRODUCTION READINESS**: Final validation before deployment

### How to Use Memory System
- Use `/mem-status` and `/mem-context` commands
- Ask "where were we in the implementation?" to get context
- Use `/mem-recent` to see recent development activity

---
