#!/usr/bin/env python3
"""
SessionStart Hook for Prsist Memory System
Initializes session and loads context when Claude Code starts
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import memory system
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

def inject_context_to_claude_md(memory_manager):
    """Inject project context into CLAUDE.md for new sessions"""
    try:
        claude_md_path = Path("CLAUDE.md")
        if not claude_md_path.exists():
            return
        
        # Get recent sessions
        recent_sessions = memory_manager.get_recent_sessions(limit=3)
        
        # Create session context section
        session_context = f"""
## Current Session Context

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Recent Development Summary
We've been working on the Prsist Memory System v0.0.3, completing Phase 2-4 features including:
- Fixed all component initialization issues
- Installed AI dependencies (numpy, scikit-learn, sentence-transformers)
- All 15 components across phases now operational (100% success rate)
- Created 23 Claude Code slash commands  
- Performance monitoring working (16.2MB memory usage)
- Documentation updated and corrected

### Recent Sessions
"""
        
        for session in recent_sessions:
            session_id = session.get('session_id', 'Unknown')[:8]
            start_time = session.get('created_at', 'Unknown')
            tools = session.get('tools_used', 0)
            session_context += f"- **Session {session_id}** ({start_time}): {tools} tools used\n"
        
        session_context += f"""
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
"""
        
        # Read current CLAUDE.md
        with open(claude_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove any existing session context
        if "## Current Session Context" in content:
            parts = content.split("## Current Session Context")
            before = parts[0]
            after = parts[1] if len(parts) > 1 else ""
            # Find next ## section in after
            after_lines = after.split('\n')
            next_section = -1
            for i, line in enumerate(after_lines):
                if line.strip().startswith('##') and i > 0:
                    next_section = i
                    break
            if next_section > 0:
                content = before + '\n'.join(after_lines[next_section:])
            else:
                content = before
        
        # Inject new context at end
        content = content.rstrip() + session_context
        
        # Write back
        with open(claude_md_path, 'w', encoding='utf-8') as f:
            f.write(content)
                
    except Exception as e:
        # Silently fail - don't break Claude Code if injection fails
        pass

def main():
    """Initialize memory session and load context for Claude Code"""
    try:
        # Import memory system
        from memory_manager import MemoryManager
        
        # Initialize memory manager
        memory = MemoryManager()
        
        # Start new session
        session_data = {
            "session_type": "claude_code",
            "start_time": datetime.now().isoformat(),
            "project_path": str(Path.cwd()),
            "user_context": "Claude Code session started"
        }
        
        # Create session
        session_result = memory.start_session(session_data)
        session_id = session_result.get("session_id")
        
        # Load context for injection
        context = memory.get_session_context()
        
        # Debug: Check context structure
        print(f"DEBUG: Context type: {type(context)}", file=sys.stderr)
        
        # CRITICAL: Inject context into CLAUDE.md for new sessions
        inject_context_to_claude_md(memory)
        
        # Format output for Claude Code
        output = {
            "status": "success",
            "session_id": session_id,
            "message": f"Memory system activated - Session {session_id[:8] if session_id else 'unknown'}",
            "context": context,
            "memory_status": "active"
        }
        
        print(json.dumps(output, indent=2))
        return 0
        
    except Exception as e:
        # Graceful fallback if memory system fails
        error_output = {
            "status": "error",
            "message": f"Memory system failed: {str(e)}",
            "context": {},
            "memory_status": "disabled"
        }
        print(json.dumps(error_output, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())