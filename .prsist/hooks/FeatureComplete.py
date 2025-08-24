#!/usr/bin/env python3
"""
Feature Complete Hook for Prsist Memory System
Tracks major feature completions and milestones instead of every tool usage
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
    """Log feature completion and create checkpoint"""
    try:
        # Import memory system
        from memory_manager import MemoryManager
        
        # Get feature information from command line args or environment
        feature_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FEATURE_NAME", "Unknown Feature")
        feature_description = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("FEATURE_DESCRIPTION", "")
        
        # Initialize memory manager
        memory = MemoryManager()
        
        # Log the feature completion
        completion_data = {
            "feature_name": feature_name,
            "feature_description": feature_description,
            "completion_time": datetime.now().isoformat(),
            "project_path": str(Path.cwd()),
            "completion_type": "feature_milestone"
        }
        
        # Update session with feature completion
        memory.update_session_context(completion_data)
        
        # Create a checkpoint for this feature
        checkpoint_name = f"feature_{feature_name.lower().replace(' ', '_')}"
        memory.create_checkpoint(checkpoint_name)
        
        # Add to project memory
        memory_entry = f"✅ **{feature_name}** completed at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if feature_description:
            memory_entry += f"\n   {feature_description}"
        
        memory.add_project_memory(memory_entry)
        
        # Return success status
        output = {
            "status": "success",
            "message": f"Feature completion logged: {feature_name}",
            "checkpoint_created": checkpoint_name,
            "feature_logged": feature_name
        }
        
        print(json.dumps(output, indent=2))
        return 0
        
    except Exception as e:
        # Graceful fallback if memory system fails
        error_output = {
            "status": "error", 
            "message": f"Feature logging failed: {str(e)}",
            "feature_logged": False
        }
        print(json.dumps(error_output, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())