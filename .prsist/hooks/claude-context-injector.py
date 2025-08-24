#!/usr/bin/env python3
"""
Claude Context Injector for Prsist Memory System
Automatically injects memory context into Claude Code sessions
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path to import memory system
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_context():
    """Get relevant project context for Claude with improved fallback"""
    try:
        from memory_manager import MemoryManager
        from context_builder import ContextBuilder
        from database import MemoryDatabase
        from utils import get_git_info
        
        # Try to get context from memory manager first
        try:
            memory = MemoryManager()
            context_text = memory.get_session_context()
            if context_text and context_text.strip():
                return context_text
        except Exception as inner_e:
            logger.warning(f"MemoryManager context failed: {inner_e}")
        
        # Fallback: Build context directly from available data
        memory_dir = Path(__file__).parent.parent
        db = MemoryDatabase(memory_dir / "storage" / "sessions.db")
        
        # Get project memory file
        project_memory_file = memory_dir / "context" / "project-memory.md"
        project_memory = ""
        if project_memory_file.exists():
            with open(project_memory_file, 'r', encoding='utf-8') as f:
                project_memory = f.read()
        
        # Get recent sessions
        recent_sessions = []
        try:
            recent_sessions = db.get_recent_sessions(limit=3)
        except:
            pass
        
        # Get git info
        git_info = get_git_info(str(Path.cwd()))
        
        # Format context for Claude injection
        claude_context = f"""# Project Context

**Project Root:** {Path.cwd()}
**Timestamp:** {datetime.now().isoformat()}
**Git Branch:** {git_info.get('branch', 'unknown')}
**Git Hash:** {git_info.get('hash', 'unknown')}

## Project Memory

{project_memory if project_memory.strip() else 'No persistent project memory yet. This will be populated as you work on the project.'}

## Recent Sessions

"""
        
        if recent_sessions:
            for session in recent_sessions:
                session_id = session.get('id', 'unknown')[:8]
                tools_used = session.get('tool_usage_count', 0)
                created_at = session.get('created_at', 'unknown')
                claude_context += f"- **Session {session_id}** ({created_at}): {tools_used} tools used\n"
        else:
            claude_context += "This is your first session or session history is not available.\n"
            
        claude_context += f"""

## Memory System Status

- **Status:** Active and ready
- **Database:** Session history available
- **Context:** Automatically maintained across Claude Code sessions

The Prsist memory system is running transparently and will track your development progress, decisions, and project evolution.
"""
        
        return claude_context
        
    except Exception as e:
        logger.error(f"All context methods failed: {e}")
        return f"""# Project Context

**Project Root:** {Path.cwd()}
**Timestamp:** {datetime.now().isoformat()}

## Memory System Status

The Prsist memory system encountered an initialization issue: {str(e)}

However, the system will continue to function and track your session data. Context will improve as you use Claude Code.

**Note:** This is an automatically generated context file that provides project memory and context for Claude Code sessions.
"""

def write_context_file():
    """Write context to the Claude context file"""
    try:
        context_dir = memory_dir / 'context'
        context_dir.mkdir(exist_ok=True)
        
        context_file = context_dir / 'claude-context.md'
        context_content = get_project_context()
        
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(context_content)
            
        logger.info(f"Claude context written to {context_file}")
        
        return {
            "status": "success",
            "context_file": str(context_file),
            "context_length": len(context_content),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to write context file: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main context injection function"""
    try:
        # Generate and write context
        result = write_context_file()
        
        # Output result for Claude Code integration
        print(json.dumps(result, indent=2))
        
        return 0 if result["status"] == "success" else 1
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "message": "Context injection failed - Claude will work without memory context",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(error_result, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())