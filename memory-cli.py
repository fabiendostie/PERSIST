#!/usr/bin/env python3
"""
Claude Code Memory CLI - Direct access to memory system
Usage: python memory-cli.py <command> [args]
"""

import sys
import subprocess
import os

def run_memory_command(command, args=None):
    """Run prsist memory command"""
    if args is None:
        args = []
    
    cmd = ['python', '.prsist/prsist.py', f'-{command}'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Available memory commands:")
        print("  status       - Show current session status")
        print("  health       - System health check")
        print("  context      - Show current context")
        print("  memory       - Memory statistics")
        print("  feature      - Log completed feature")
        print("  decision     - Add decision")
        print("  checkpoint   - Create checkpoint")
        print("  recent       - Show recent sessions")
        print("  test         - Run system test")
        print("  validate     - Validate system")
        print("")
        print("Usage: python memory-cli.py <command>")
        return 1
    
    command = sys.argv[1].lower()
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Map friendly names to prsist commands
    command_map = {
        'status': 's',
        'health': 'h', 
        'context': 'c',
        'memory': 'm',
        'feature': 'f',
        'decision': 'd',
        'checkpoint': 'k',
        'recent': 'r',
        'test': 't',
        'validate': 'v'
    }
    
    if command in command_map:
        return run_memory_command(command_map[command], args)
    else:
        print(f"Unknown command: {command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())