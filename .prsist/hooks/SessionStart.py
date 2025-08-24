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