#!/usr/bin/env python3
"""
PostToolUse Hook for Prsist Memory System
Tracks tool usage and updates memory after each tool execution
"""

import sys
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import memory system
memory_dir = Path(__file__).parent.parent
sys.path.insert(0, str(memory_dir))

def main():
    """Log tool usage and update memory system"""
    try:
        # Import memory system
        from memory_manager import MemoryManager
        
        # Get hook data from stdin (Claude Code sends JSON)
        hook_data = {}
        if not sys.stdin.isatty():
            try:
                stdin_content = sys.stdin.read().strip()
                if stdin_content:
                    hook_data = json.loads(stdin_content)
            except json.JSONDecodeError as e:
                # Log the error but don't fail
                hook_data = {"error": f"JSON decode error: {e}"}
        
        # Extract tool information from Claude Code hook data
        tool_name = hook_data.get("tool_name", "unknown")
        tool_input = hook_data.get("tool_input", {})
        tool_response = hook_data.get("tool_response", {})
        session_id = hook_data.get("session_id", "unknown")
        cwd = hook_data.get("cwd", os.getcwd())
        
        # Determine success from tool_response
        success = True
        if isinstance(tool_response, dict):
            success = not bool(tool_response.get("error"))
        
        # Extract file path for file operations
        file_path = None
        if isinstance(tool_input, dict):
            file_path = tool_input.get("file_path") or tool_input.get("notebook_path")
        
        # Initialize memory manager
        memory = MemoryManager()
        
        # Log tool usage with Claude Code data
        memory.log_tool_usage(
            tool_name=tool_name,
            input_data=tool_input,
            output_data=str(tool_response),
            success=success,
            execution_time_ms=0  # Claude Code doesn't provide timing
        )
        
        # Log file interaction if this was a file operation
        if file_path and tool_name in ["Write", "Edit", "MultiEdit", "NotebookEdit"]:
            memory.log_file_interaction(
                file_path=file_path,
                action_type=tool_name.lower(),
                line_changes=None  # Could be enhanced to track actual changes
            )
        
        # Update session with tool usage information
        context_updates = {
            "last_tool_used": tool_name,
            "last_tool_time": datetime.now().isoformat(),
            "tool_count": 1  # This will be aggregated by session tracker
        }
        
        memory.update_session_context(context_updates)
        
        # Optional: Create checkpoint for significant tools
        significant_tools = ["Write", "Edit", "MultiEdit", "NotebookEdit"]
        if tool_name in significant_tools:
            memory.create_checkpoint(f"auto_checkpoint_{tool_name.lower()}")
        
        # Update Claude context file automatically with improved context after activity
        try:
            # Delayed context update after session has some activity
            subprocess.run([
                sys.executable, 
                str(memory_dir / 'hooks' / 'claude-context-injector.py')
            ], capture_output=True, timeout=5)
        except:
            pass  # Silent failure to maintain transparency
        
        # Return success status (minimal for transparency)
        output = {
            "status": "success",
            "message": f"Memory updated",
            "session_updated": True,
            "tool_logged": tool_name
        }
        
        print(json.dumps(output, indent=2))
        return 0
        
    except Exception as e:
        # Graceful fallback if memory system fails
        error_output = {
            "status": "error", 
            "message": f"Memory update failed: {str(e)}",
            "session_updated": False
        }
        print(json.dumps(error_output, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())