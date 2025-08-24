#!/usr/bin/env python3
"""
Debug version of PostToolUse Hook
Logs all input to help diagnose issues
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

def main():
    """Debug hook - log everything we receive"""
    
    # Log to a debug file
    debug_file = Path(__file__).parent.parent / "debug-hook-calls.log"
    
    with open(debug_file, "a") as f:
        f.write(f"\n=== PostToolUse Hook Called at {datetime.now()} ===\n")
        
        # Log environment variables
        f.write("Environment variables:\n")
        for key, value in os.environ.items():
            if 'CLAUDE' in key.upper():
                f.write(f"  {key} = {value}\n")
        
        # Log stdin content
        f.write("\nSTDIN content:\n")
        if not sys.stdin.isatty():
            stdin_content = sys.stdin.read()
            f.write(f"Raw: {stdin_content}\n")
            
            try:
                data = json.loads(stdin_content) if stdin_content.strip() else {}
                f.write(f"Parsed JSON: {json.dumps(data, indent=2)}\n")
            except json.JSONDecodeError as e:
                f.write(f"JSON Parse Error: {e}\n")
        else:
            f.write("No stdin data (TTY mode)\n")
        
        # Log command line args
        f.write(f"\nCommand line args: {sys.argv}\n")
        f.write("=== End Hook Call ===\n\n")
    
    # Return success
    print(json.dumps({"status": "success", "message": "Debug hook executed"}))
    return 0

if __name__ == "__main__":
    sys.exit(main())